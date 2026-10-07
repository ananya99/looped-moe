"""Parameter tagging + rule-based optimizer construction + LR schedule.

Every parameter is tagged once with a *role* derived from the module that owns it
(not from substring matches on its name):

    embed   nn.Embedding                   hidden  attention projections (2D)
    head    the output nn.Linear           expert  GroupedExperts 3D tensors
    router  Router.weight                  norm    RMSNorm weights (1D)

An ordered list of `Rule`s maps roles to (optimizer, lr multiplier, weight decay).
The defaults reproduce the original trainer's table:

    | role                  | optimizer | lr        | wd   |
    |-----------------------|-----------|-----------|------|
    | expert, hidden        | Muon      | lr        | 0.1  |
    | router, embed, head,  | AdamW     | lr * 0.1  | 0.01 |
    | norm                  |           |           |      |

A new parameter-group idea (different LR for routers, separate decay for experts,
per-expert anything) is a new rule or a new group attribute, not a code path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..config import OptimConfig
from ..model import Router, GroupedExperts, RMSNorm, Transformer
from .muon import Muon


@dataclass
class ParamInfo:
    name: str
    param: nn.Parameter
    role: str
    layer: int | None


def tag_parameters(model: Transformer) -> list[ParamInfo]:
    out, seen = [], set()
    for mod_name, mod in model.named_modules():
        layer = None
        parts = mod_name.split(".")
        if len(parts) > 1 and parts[0] == "layers":
            layer = int(parts[1])
        for p_name, p in mod.named_parameters(recurse=False):
            if id(p) in seen:
                continue
            seen.add(id(p))
            if isinstance(mod, nn.Embedding):
                role = "embed"
            elif mod is model.output:
                role = "head"
            elif isinstance(mod, Router):
                role = "router"
            elif isinstance(mod, GroupedExperts):
                role = "expert"
            elif isinstance(mod, RMSNorm):
                role = "norm"
            elif isinstance(mod, nn.Linear):
                role = "hidden"
            else:
                raise ValueError(f"No role for parameter {mod_name}.{p_name}; add one in tag_parameters")
            full = f"{mod_name}.{p_name}" if mod_name else p_name
            out.append(ParamInfo(full, p, role, layer))
    n_model = sum(1 for _ in model.parameters())
    assert len(out) == n_model, f"tagged {len(out)} of {n_model} parameters"
    return out


@dataclass
class Rule:
    roles: tuple[str, ...]
    optimizer: str            # "muon" | "adamw"
    lr_mult: float = 1.0
    weight_decay: float = 0.0


def default_rules(name: str) -> list[Rule]:
    if name == "muon":
        return [Rule(("expert", "hidden"), "muon", 1.0, 0.1),
                Rule(("router", "embed", "head", "norm"), "adamw", 0.1, 0.01)]
    if name == "adamw":
        # Original `optimizer=adamw` path: one AdamW over everything at the full lr.
        return [Rule(("expert", "hidden", "router", "embed", "head", "norm"), "adamw", 1.0, 0.01)]
    raise ValueError(f"unknown optimizer {name!r}")


class Optimizers:
    """Several torch optimizers behind one interface. Each param group carries
    `lr_mult`; `set_lr(base)` sets group lr = base * lr_mult."""

    def __init__(self, optimizers: dict[str, torch.optim.Optimizer]):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        return [g for o in self.optimizers.values() for g in o.param_groups]

    def set_lr(self, base_lr: float):
        for g in self.param_groups:
            g["lr"] = base_lr * g["lr_mult"]

    def zero_grad(self, set_to_none: bool = True):
        for o in self.optimizers.values():
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers.values():
            o.step()

    def state_dict(self):
        return {k: o.state_dict() for k, o in self.optimizers.items()}

    def load_state_dict(self, sd):
        assert set(sd) == set(self.optimizers), (set(sd), set(self.optimizers))
        for k, o in self.optimizers.items():
            o.load_state_dict(sd[k])


def build_optimizers(model: Transformer, cfg: OptimConfig, rules: list[Rule] | None = None,
                     verbose: bool = False) -> Optimizers:
    rules = rules or default_rules(cfg.name)
    infos = tag_parameters(model)
    groups: dict[str, list[dict]] = {}
    claimed = set()
    for rule in rules:
        params = [i.param for i in infos if i.role in rule.roles and id(i.param) not in claimed]
        claimed.update(id(p) for p in params)
        if params:
            groups.setdefault(rule.optimizer, []).append(dict(
                params=params, lr_mult=rule.lr_mult, weight_decay=rule.weight_decay,
                roles=rule.roles))
    missing = [i.name for i in infos if id(i.param) not in claimed]
    if missing:
        raise ValueError(f"parameters not covered by any rule: {missing[:5]} ...")

    optimizers = {}
    for name, gs in groups.items():
        for g in gs:
            g["lr"] = cfg.lr * g["lr_mult"]
        if name == "muon":
            optimizers[name] = Muon(gs, lr=cfg.lr, momentum=cfg.muon_momentum,
                                    nesterov=cfg.muon_nesterov)
        elif name == "adamw":
            optimizers[name] = torch.optim.AdamW(gs, lr=cfg.lr, betas=tuple(cfg.adamw_betas))
        else:
            raise ValueError(name)

    if verbose:
        for name, o in optimizers.items():
            for g in o.param_groups:
                n = sum(p.numel() for p in g["params"])
                print(f"  {name:6s} roles={','.join(g['roles']):28s} lr_mult={g['lr_mult']:<5g} "
                      f"wd={g['weight_decay']:<5g} params={n:,}")
    return Optimizers(optimizers)


def lr_at(step: int, cfg: OptimConfig, n_steps: int) -> float:
    """Base LR (before lr_mult): linear warmup then cosine to lr * min_lr_ratio.
    Same curve as the original `get_lr`, applied to every group via lr_mult."""
    lr, min_lr = cfg.lr, cfg.lr * cfg.min_lr_ratio
    if step < cfg.warmup_steps:
        return lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, n_steps - cfg.warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))
