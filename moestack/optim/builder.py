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
from contextlib import contextmanager
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
    # One param group per parameter instead of one per rule. Groups then carry the
    # owning layer index, which is how a per-layer policy (per-expert LR) addresses
    # them. Costs nothing: an optimizer loops over params inside a group anyway.
    per_param_groups: bool = False


def default_rules(name: str, split_experts: bool = False) -> list[Rule]:
    """The original trainer's routing table. `split_experts` puts each layer's
    expert tensors in their own groups so a callback can set per-layer attributes
    on them; the arithmetic is unchanged."""
    if name == "muon":
        if split_experts:
            return [Rule(("expert",), "muon", 1.0, 0.1, per_param_groups=True),
                    Rule(("hidden",), "muon", 1.0, 0.1),
                    Rule(("router", "embed", "head", "norm"), "adamw", 0.1, 0.01)]
        return [Rule(("expert", "hidden"), "muon", 1.0, 0.1),
                Rule(("router", "embed", "head", "norm"), "adamw", 0.1, 0.01)]
    if name == "adamw":
        # Original `optimizer=adamw` path: one AdamW over everything at the full lr.
        # Experts stay 3D single tensors here, so there is nothing to address
        # per expert -- see Optimizers.set_expert_lr_scales.
        return [Rule(("expert", "hidden", "router", "embed", "head", "norm"), "adamw", 1.0, 0.01)]
    raise ValueError(f"unknown optimizer {name!r}")


@contextmanager
def _hidden_grads(params):
    """Temporarily detach `.grad` from `params` so an optimizer step skips them.

    Both Muon and AdamW skip a parameter whose grad is None, so this is how a
    partial step leaves the rest of the model alone *without clearing* what it has
    accumulated. That distinction is the whole point: once `.grad` is being used as
    a multi-batch accumulator, zeroing it to skip a parameter would throw away the
    window.
    """
    saved = [(p, p.grad) for p in params if p.grad is not None]
    for p, _ in saved:
        p.grad = None
    try:
        yield
    finally:
        for p, g in saved:
            p.grad = g


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

    def expert_lr_groups(self) -> dict[int, list[dict]]:
        """layer index -> the param groups holding that layer's expert tensors.

        Empty unless the rules were built with `split_experts=True`; a group that
        mixes layers has no single layer to key on and is skipped.
        """
        out: dict[int, list[dict]] = {}
        for g in self.param_groups:
            if "expert" in g.get("roles", ()) and g.get("layer") is not None:
                out.setdefault(g["layer"], []).append(g)
        return out

    def _set_expert_attr(self, key: str, values, what: str):
        """Write one row per layer of a `[L, E]` tensor onto that layer's expert
        groups. Both of a layer's expert tensors (gate_up and down) get the same
        row. These are *attributes*, not a modified `lr`, so they cannot compound
        across steps the way `pg["lr"] *= scale` could. `values=None` clears the
        attribute, which restores the unmodified code path inside Muon.
        """
        groups = self.expert_lr_groups()
        if not groups:
            raise RuntimeError(
                f"no per-layer expert param groups: {what} needs "
                "optim.name=muon and rules built with split_experts=True "
                "(under AdamW the experts stay single 3D tensors and are not "
                "individually addressable)")
        if values is not None and (values.dim() != 2 or values.size(0) <= max(groups)):
            raise ValueError(f"expected {key} [n_layers, n_experts] covering layers "
                             f"0..{max(groups)}, got {tuple(values.shape)}")
        for layer, gs in groups.items():
            for g in gs:
                g[key] = None if values is None else values[layer]

    def set_expert_lr_scales(self, scales):
        """Per-expert LR multipliers, a `[L, E]` tensor. See `_set_expert_attr`."""
        self._set_expert_attr("lr_scale", scales, "per-expert learning rates")

    def set_expert_update_mask(self, mask):
        """Which experts take a step at all, a `[L, E]` bool tensor (None = all).

        Unlike an `lr_scale` of 0, a False entry also freezes that expert's momentum
        buffer, which is what lets an expert accumulate gradient over several
        batches without its buffer decaying while it waits.
        """
        if mask is not None and mask.dtype != torch.bool:
            raise ValueError(f"update mask must be a bool tensor, got {mask.dtype}")
        self._set_expert_attr("update_mask", mask, "per-expert update masks")

    def zero_grad(self, set_to_none: bool = True, keep=()):
        """`keep` names parameters whose `.grad` is a multi-batch accumulator rather
        than this batch's gradient (moestack/expert_accum.py), and so must survive
        the clear. Empty by default, which is the original behaviour."""
        with _hidden_grads(keep):
            for o in self.optimizers.values():
                o.zero_grad(set_to_none=set_to_none)

    def step(self, only: str | None = None):
        """`only="expert"` steps just the groups holding expert tensors and leaves
        every other parameter's accumulated gradient exactly where it is -- which is
        what a mid-batch expert event needs, since the shared parameters are still
        halfway through their own accumulation window."""
        hidden = ([] if only is None else
                  [p for g in self.param_groups if only not in g.get("roles", ())
                   for p in g["params"]])
        with _hidden_grads(hidden):
            for o in self.optimizers.values():
                o.step()

    def state_dict(self):
        return {k: o.state_dict() for k, o in self.optimizers.items()}

    def load_state_dict(self, sd):
        assert set(sd) == set(self.optimizers), (set(sd), set(self.optimizers))
        for k, o in self.optimizers.items():
            n_now, n_ckpt = len(o.param_groups), len(sd[k]["param_groups"])
            if n_now != n_ckpt:
                # torch's own message ("different number of parameter groups") does
                # not say why. In practice there is one cause: the checkpoint was
                # written with a different optim.per_expert_lr setting, which moves
                # the expert tensors into (or out of) per-layer groups.
                raise ValueError(
                    f"checkpoint's {k} optimizer has {n_ckpt} param groups but this "
                    f"run builds {n_now}: the checkpoint was written with a different "
                    f"optim.per_expert_lr setting (enabling it splits the expert "
                    f"tensors into one group per layer). Resume with the configuration "
                    f"the checkpoint was written with.")
            o.load_state_dict(sd[k])


def build_optimizers(model: Transformer, cfg: OptimConfig, rules: list[Rule] | None = None,
                     verbose: bool = False, split_experts: bool = False) -> Optimizers:
    rules = rules or default_rules(cfg.name, split_experts=split_experts)
    infos = tag_parameters(model)
    groups: dict[str, list[dict]] = {}
    claimed = set()
    for rule in rules:
        matched = [i for i in infos if i.role in rule.roles and id(i.param) not in claimed]
        claimed.update(id(i.param) for i in matched)
        base = dict(lr_mult=rule.lr_mult, weight_decay=rule.weight_decay, roles=rule.roles)
        if rule.per_param_groups:
            groups.setdefault(rule.optimizer, []).extend(
                dict(base, params=[i.param], layer=i.layer, name=i.name) for i in matched)
        elif matched:
            groups.setdefault(rule.optimizer, []).append(dict(base, params=[i.param for i in matched]))
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
            # Per-param rules make one group per tensor; report them as one line
            # per (roles, lr_mult, wd) with the group count, not 32 near-copies.
            agg: dict[tuple, list[int]] = {}
            for g in o.param_groups:
                key = (",".join(g["roles"]), g["lr_mult"], g["weight_decay"])
                n_el, n_g = agg.setdefault(key, [0, 0])
                agg[key] = [n_el + sum(p.numel() for p in g["params"]), n_g + 1]
            for (roles, lr_mult, wd), (n_el, n_g) in agg.items():
                extra = f" groups={n_g}" if n_g > 1 else ""
                print(f"  {name:6s} roles={roles:28s} lr_mult={lr_mult:<5g} "
                      f"wd={wd:<5g} params={n_el:,}{extra}")
    return Optimizers(optimizers)


def lr_at(step: int, cfg: OptimConfig, n_steps: int) -> float:
    """Base LR (before lr_mult): linear warmup then cosine to lr * min_lr_ratio.
    Same curve as the original `get_lr`, applied to every group via lr_mult."""
    lr, min_lr = cfg.lr, cfg.lr * cfg.min_lr_ratio
    if step < cfg.warmup_steps:
        return lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, n_steps - cfg.warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))
