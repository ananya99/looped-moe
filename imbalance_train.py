#!/usr/bin/env python3
"""
imbalance_train.py — Baseline MoE pretraining + per-expert learning-rate experiment
==================================================================================

A standalone trainer for the expert-load imbalance experiment:

    How does the number of tokens routed to an expert influence the learning of
    that expert's weights and of its router row? Can the learning rate be varied
    per expert based on it?

This is `moe_train.py` with the expert-tying machinery removed. Tying is what the
paper is about, but it is entirely inert at `--tie-group-size 1`, so carrying it
here only obscures the baseline. Dropped relative to `moe_train.py`:

  * `tie_expert_layers()` and the flags `--tie-group-size`, `--tie-skip-first/last`,
    `--expand-tied-experts`, `--tied-lr-divisor`
  * the `cross_loop_agreement` / `routing_diversity` metrics, which only mean
    something when several layers share one expert tensor
  * the `router/*` dashboard metrics, which were top-1, last-micro-batch and
    rank-0; the exact global-batch equivalents (`load/*`) come from
    `expert_stats.ExpertLoadTracker`

Kept, because they apply to the baseline and not just to tying: the router z-loss
patch, the depth-scaled Xavier re-init, the Muon 2D-proxy split for 3D expert
tensors, DDP, bf16 autocast, gradient checkpointing, and checkpoint resume.

What is new here: every expert is *always* its own Muon parameter group, so a
per-expert learning rate is a dict assignment rather than a code path. At
`--per-expert-lr-alpha 0` without `--freeze-dead-experts` every multiplier is
exactly 1.0 and this reduces to the plain baseline.

Checkpoints are readable by `eval_downstream.py --tie-group-size 1`, but their
optimizer state is NOT interchangeable with `moe_train.py`'s: the Muon parameter
groups are shaped differently.

Launch (1 GPU):
  python imbalance_train.py --arch deepseek --scale tiny --aux-loss-coef 0 --n-steps 2000
Launch (4 GPUs DDP):
  torchrun --standalone --nproc_per_node=4 imbalance_train.py --arch deepseek --aux-loss-coef 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import hydra
from omegaconf import DictConfig, OmegaConf

from data import get_dataloader, get_eval_batches
from run_tracking import build_run_name, init_wandb
from expert_stats import (
    ExpertLoadTracker,
    ExpertStatsWriter,
    compute_lr_scales,
    find_layers,
    gini,
    spearman,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("wandb not installed — logging to console only")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

torch.set_float32_matmul_precision("high")

# HF multiplies the entire router-loss return value by config.router_aux_loss_coef.
# We keep that at a fixed non-zero carrier and pre-divide each term by it, so the
# load-balancing and z-loss coefficients are independent and either can be zero.
AUX_CARRIER = 0.01

MUON_WD = 0.1    # hidden 2D/3D weights (production MoE convention)
ADAMW_WD = 0.01  # embeddings, head, routers, 1D (Keller Jordan reference recipe)


# ──────────────────────────────────────────────────────────────────────
# 1. Architecture configs
# ──────────────────────────────────────────────────────────────────────
# Widths scale with --scale; layer counts are fixed per architecture.
# router_aux_loss_coef is the carrier described above, NOT the effective
# load-balancing strength; use --aux-loss-coef for that.

def get_olmoe_config(scale="regular"):
    """OLMoE-1B-7B (Allen AI, Sep 2024) — arxiv.org/abs/2409.02060"""
    from transformers import OlmoeConfig
    small, tiny = scale == "small", scale == "tiny"
    return OlmoeConfig(
        vocab_size=100277,
        hidden_size=512 if tiny else (1024 if small else 2048),
        intermediate_size=256 if tiny else (512 if small else 1024),
        num_hidden_layers=16,
        num_attention_heads=4 if tiny else (8 if small else 16),
        num_key_value_heads=4 if tiny else (8 if small else 16),
        num_experts=64,
        num_experts_per_tok=8,
        router_aux_loss_coef=AUX_CARRIER,
        max_position_embeddings=4096,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        clip_qkv=8.0,
        tie_word_embeddings=False,
        use_cache=False,
        output_router_logits=True,
    )


def get_qwen3moe_config(scale="regular"):
    """Qwen3-MoE-style (Qwen Team, Apr 2025) — arxiv.org/abs/2505.09388"""
    from transformers import Qwen3MoeConfig
    small, tiny = scale == "small", scale == "tiny"
    return Qwen3MoeConfig(
        vocab_size=100277,
        hidden_size=384 if tiny else (768 if small else 1536),
        intermediate_size=1024 if tiny else (2048 if small else 4096),
        moe_intermediate_size=192 if tiny else (384 if small else 768),
        num_hidden_layers=28,
        num_attention_heads=6 if tiny else (12 if small else 24),
        num_key_value_heads=1 if tiny else (2 if small else 4),
        num_experts=60,
        num_experts_per_tok=4,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        router_aux_loss_coef=AUX_CARRIER,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        use_cache=False,
        output_router_logits=True,
    )


def get_deepseek_config(scale="regular"):
    """DeepSeekMoE-*style*: OlmoeConfig with top-6 routing.

    No MLA, no fine-grained experts, no always-on shared expert — the name is
    inherited from moe_train.py and is a misnomer.
    """
    from transformers import OlmoeConfig
    small, tiny = scale == "small", scale == "tiny"
    return OlmoeConfig(
        vocab_size=100277,
        hidden_size=512 if tiny else (1024 if small else 2048),
        intermediate_size=256 if tiny else (512 if small else 1024),
        num_hidden_layers=16,
        num_attention_heads=4 if tiny else (8 if small else 16),
        num_key_value_heads=4 if tiny else (8 if small else 16),
        num_experts=64,
        num_experts_per_tok=6,
        router_aux_loss_coef=AUX_CARRIER,
        max_position_embeddings=4096,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        clip_qkv=8.0,
        tie_word_embeddings=False,
        use_cache=False,
        output_router_logits=True,
    )


ARCH_REGISTRY = {
    "olmoe": ("OLMoE-1B-7B", get_olmoe_config),
    "qwen3moe": ("Qwen3-MoE-style", get_qwen3moe_config),
    "deepseek": ("DeepSeekMoE-style", get_deepseek_config),
}

TRAIN_CONFIG = dict(
    batch_size=16, grad_accum=4, seq_len=2048,
    n_steps=20_000, lr=2e-2, min_lr=2e-3,
    warmup_steps=100, grad_clip=1.0,
    eval_every=500, eval_batches=10, save_every=500,
    log_every=10,
)


# ──────────────────────────────────────────────────────────────────────
# 2. Router loss: load balancing + z-loss
# ──────────────────────────────────────────────────────────────────────

def install_router_loss_patch(aux_coef, z_coef, master_process=True):
    """Replace HF's load_balancing_loss_func with one that adds a router z-loss.

    HF computes `loss = lm_loss + config.router_aux_loss_coef * f(gate_logits)`.
    Pre-dividing each term inside f by that carrier makes the effective
    coefficients exactly `aux_coef` and `z_coef`, independently — which is how
    load balancing can be switched off while the z-loss stays on.

    Asserts that at least one patch site existed, so a transformers upgrade that
    moves the symbol fails loudly instead of silently reverting to HF's stock
    implementation (which has no z-loss term).
    """
    aux_scale = aux_coef / AUX_CARRIER
    z_scale = z_coef / AUX_CARRIER

    def patched(gate_logits, num_experts, top_k, attention_mask=None):
        if gate_logits is None or len(gate_logits) == 0:
            return 0.0
        total, n_layers = 0.0, 0
        for layer_logits in gate_logits:
            if layer_logits is None:
                continue
            if isinstance(layer_logits, tuple):
                layer_logits = layer_logits[0]
            n_exp = layer_logits.size(-1)
            flat = layer_logits.view(-1, n_exp)

            # float32 before softmax/logsumexp: bf16 overflows here.
            f32 = flat.float()
            probs = torch.nn.functional.softmax(f32, dim=-1).type_as(flat)

            # Router z-loss (Zoph et al., 2022): penalises large logits.
            z_loss = torch.mean(torch.logsumexp(f32, dim=-1) ** 2) * z_scale

            _, selected = torch.topk(probs, top_k, dim=-1)
            mask = torch.nn.functional.one_hot(selected, n_exp)
            # /top_k so the frequencies sum to 1 (cf. HF #43688).
            tokens_per_expert = mask.sum(dim=1).float().mean(dim=0) / top_k
            prob_per_expert = probs.mean(dim=0)
            # Load balancing (Shazeer 2017; Fedus 2022).
            balance = (tokens_per_expert * prob_per_expert).sum() * n_exp

            total = total + aux_scale * balance + z_loss
            n_layers += 1
        return total / n_layers if n_layers else 0.0

    patched_sites = []
    for modpath in ("transformers.models.olmoe.modeling_olmoe",
                    "transformers.models.qwen3_moe.modeling_qwen3_moe",
                    "transformers.loss.loss_utils",
                    "transformers.modeling_utils"):
        try:
            mod = __import__(modpath, fromlist=["load_balancing_loss_func"])
        except ImportError:
            continue
        if hasattr(mod, "load_balancing_loss_func"):
            mod.load_balancing_loss_func = patched
            patched_sites.append(modpath)

    assert patched_sites, (
        "load_balancing_loss_func not found in any known HF location — the "
        "transformers API has changed and this patch is dead."
    )
    if master_process:
        print(f"  Router loss patched in: {patched_sites}")
        print(f"    effective aux_coef={aux_coef}, z_loss_coef={z_coef}")


# ──────────────────────────────────────────────────────────────────────
# 3. Optimizer
# ──────────────────────────────────────────────────────────────────────

def build_optimizer(model, optimizer_type, lr, master_process=True):
    """Muon for hidden 2D/3D weights, AdamW for embeddings, head, routers and 1D.

    Muon's Newton-Schulz orthogonalisation cannot act on a 3D `[E, out, in]`
    expert tensor (HF's grouped_mm layout). Each expert's `[out, in]` slice is
    therefore wrapped in a 2D proxy Parameter that *shares storage* with the
    slice, and `DualOptimizer._sync_expert_grads()` copies `grad[i]` into the
    proxy before each step.

    Every proxy gets its own parameter group, tagged with `expert_layer` and
    `expert_index`. That is what makes a per-expert learning rate possible: an
    optimizer stores `lr` per group, not per parameter. It is free — Muon's
    `_single_tensor_muon` is a Python loop over individual parameters and does
    not support `foreach`, so N groups of one parameter cost the same as one
    group of N.

    Returns (optimizer, expert_lr_groups) where expert_lr_groups maps
    (layer_idx, expert_idx) -> list of that expert's parameter groups (one for
    the gate_up slice, one for the down slice).
    """
    if optimizer_type == "adamw":
        # Experts stay as single 3D tensors here, so there is nothing per-expert
        # to address; per-expert LR is rejected in main() for this path.
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=ADAMW_WD, betas=(0.9, 0.95),
        ), {}

    if optimizer_type != "muon":
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    muon_hidden = []      # 2D attention / MLP weights
    adamw_params = []     # embeddings, head, routers, norms
    expert_proxies = []   # (param_3d, [proxy per expert])
    per_expert_groups = []

    embed_head_keywords = {"embed_tokens", "lm_head", "wte", "wpe", "embed", "head"}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_embed_head = any(kw in name for kw in embed_head_keywords)
        is_router = "gate" in name and "weight" in name and p.dim() == 2

        if is_embed_head or is_router:
            adamw_params.append(p)
        elif p.dim() == 2:
            muon_hidden.append(p)
        elif p.dim() == 3:
            layer_idx = -1
            try:
                parts = name.split(".")
                if "layers" in parts:
                    layer_idx = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                pass

            proxies = []
            for i in range(p.shape[0]):
                proxy = torch.nn.Parameter(p.data[i], requires_grad=False)
                proxies.append(proxy)
                per_expert_groups.append({
                    "params": [proxy],
                    "expert_layer": layer_idx,
                    "expert_index": i,
                    # Explicit, because setting lr=0 for a frozen expert must
                    # also suppress its decay: Muon does param.mul_(1 - lr*wd).
                    "weight_decay": MUON_WD,
                })
            expert_proxies.append((p, proxies))
        else:
            adamw_params.append(p)

    from torch.optim import Muon as TorchMuon

    muon_opt = TorchMuon(
        [{"params": muon_hidden, "weight_decay": MUON_WD}] + per_expert_groups,
        lr=lr, momentum=0.95, nesterov=True, weight_decay=MUON_WD,
    )
    adamw_opt = torch.optim.AdamW(
        adamw_params, lr=lr * 0.1, weight_decay=ADAMW_WD, betas=(0.9, 0.95),
    )

    expert_lr_groups = {}
    for pg in muon_opt.param_groups:
        if "expert_layer" in pg:
            expert_lr_groups.setdefault((pg["expert_layer"], pg["expert_index"]), []).append(pg)

    if master_process:
        n_expert = sum(q.numel() for g in per_expert_groups for q in g["params"])
        print(f"  Muon: {sum(q.numel() for q in muon_hidden):,} hidden "
              f"+ {n_expert:,} expert params "
              f"({len(per_expert_groups)} single-parameter expert groups, "
              f"{len(expert_lr_groups)} experts addressable)")
        print(f"  AdamW: {sum(q.numel() for q in adamw_params):,} params")

    return DualOptimizer(muon_opt, adamw_opt, expert_proxies), expert_lr_groups


class DualOptimizer:
    """Steps Muon and AdamW together behind one interface."""

    def __init__(self, muon, adamw, expert_proxies):
        self.muon = muon
        self.adamw = adamw
        self.expert_proxies = expert_proxies
        self.param_groups = muon.param_groups + adamw.param_groups

    def zero_grad(self, set_to_none=False):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)
        # The 3D expert tensors belong to no optimizer (they are mirrored through
        # proxies), so their grads must be cleared explicitly.
        for param_3d, _ in self.expert_proxies:
            if param_3d.grad is not None:
                param_3d.grad = None if set_to_none else param_3d.grad.zero_()

    def _sync_expert_grads(self):
        # Unconditional: an expert that received no tokens still needs a (zero)
        # gradient slot so decoupled weight decay applies uniformly.
        for param_3d, proxies in self.expert_proxies:
            if param_3d.grad is None:
                continue
            for i, proxy in enumerate(proxies):
                proxy.grad = param_3d.grad[i]

    def step(self):
        self._sync_expert_grads()
        self.muon.step()
        self.adamw.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, sd):
        self.muon.load_state_dict(sd["muon"])
        self.adamw.load_state_dict(sd["adamw"])


def get_lr(step, warmup_steps, n_steps, lr, min_lr):
    """Linear warmup, then cosine decay to min_lr."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


# ──────────────────────────────────────────────────────────────────────
# 4. Init, verification, eval
# ──────────────────────────────────────────────────────────────────────

def reinit_weights_depth_scaled(model, config, master_process=True):
    """xavier_uniform(gain=1/sqrt(3*depth)) in place of HF's trunc_normal(0.02)."""
    gain = 1.0 / math.sqrt(3 * config.num_hidden_layers)
    n = 0
    for _, p in model.named_parameters():
        if p.dim() == 2:
            nn.init.xavier_uniform_(p, gain=gain)
            n += 1
        elif p.dim() == 3:
            for i in range(p.shape[0]):
                nn.init.xavier_uniform_(p[i], gain=gain)
            n += 1
        # 1D params (norms, biases) keep their default init.
    if master_process:
        print(f"  Depth-scaled Xavier init: gain={gain:.4f}, {n} tensors")


def verify_grouped_mm(model, config, master_process=True):
    if not master_process:
        return
    impl = getattr(config, "_experts_implementation", "unknown")
    has_gmm = hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm")
    try:
        experts = find_layers(model)[0].mlp.experts
        shape = tuple(experts.gate_up_proj.shape)
    except AttributeError:
        shape = None
    print(f"  experts_implementation={impl}, grouped_mm available={has_gmm}, "
          f"gate_up_proj={shape}")
    if impl == "grouped_mm" and has_gmm:
        print("  → fused grouped GEMM active")
    else:
        print("  → NOT using fused grouped GEMM; expect slower expert dispatch")


@torch.no_grad()
def evaluate(model, eval_batches, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    use_amp = device.type == "cuda"
    for input_ids, targets in eval_batches:
        input_ids, targets = input_ids.to(device), targets.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            # No router logits: HF then skips the aux-loss computation entirely.
            out = model(input_ids=input_ids, labels=input_ids, output_router_logits=False)
        total_loss += out.loss.item() * targets.numel()
        total_tokens += targets.numel()
    model.train()
    avg = total_loss / total_tokens
    return avg, math.exp(min(avg, 20))


# ──────────────────────────────────────────────────────────────────────
# 5. Per-expert statistics
# ──────────────────────────────────────────────────────────────────────

def expert_drift_stats(step, tracker, expert_tensors, w_before, init_weights, writer=None):
    """Per-expert learning statistics for one optimizer step.

    For each (layer, expert): tokens routed, gradient norm of that expert's own
    weight slices, the norm of the update actually applied, current weight norm,
    optional drift from initialisation, router row norm and mean router logit.

    The experiment's deliverable is the relationship between token share and how
    far an expert actually moves, so the per-layer rank correlation is returned
    for wandb and the full joint distribution goes to the JSONL writer.

    Gradients are read after step() but before the next zero_grad(), so they are
    the gradients that produced this step — post-clipping, as the step was.
    """
    counts = tracker.counts()
    mean_logits = tracker.mean_logits()
    corr_update, corr_grad, corr_update_alive = [], [], []

    for li in sorted(counts):
        tensors = expert_tensors.get(li)
        if tensors is None:
            continue
        c = counts[li].float()
        E = c.numel()
        gate_up, down, router = tensors["gate_up"], tensors["down"], tensors["router"]
        if gate_up.size(0) != E:
            continue

        share = c / c.mean().clamp(min=1e-9)

        grad_norm = torch.zeros(E, device=c.device)
        for t in (gate_up, down):
            if t.grad is not None:
                grad_norm += t.grad.detach().float().flatten(1).norm(dim=1)

        w_norm = sum(t.detach().float().flatten(1).norm(dim=1) for t in (gate_up, down))

        update_norm = torch.zeros(E, device=c.device)
        if w_before.get(li):
            for key, t in (("gate_up", gate_up), ("down", down)):
                update_norm += (t.detach().float() - w_before[li][key].float()).flatten(1).norm(dim=1)

        drift = torch.zeros(E, device=c.device)
        if init_weights.get(li):
            for key, t in (("gate_up", gate_up), ("down", down)):
                drift += (t.detach().float() - init_weights[li][key].float()).flatten(1).norm(dim=1)

        router_norm = (router.detach().float().norm(dim=1) if router.dim() == 2
                       else torch.zeros(E, device=c.device))
        mlogit = mean_logits.get(li, torch.zeros(E, device=c.device))

        alive = c > 0
        if w_before.get(li):
            corr_update.append(spearman(share, update_norm))
            if int(alive.sum().item()) >= 3:
                corr_update_alive.append(spearman(share[alive], update_norm[alive]))
        corr_grad.append(spearman(share, grad_norm))

        if writer is not None:
            writer.write({
                "step": step,
                "layer": li,
                "counts": c.int().tolist(),
                "share": [round(v, 5) for v in share.tolist()],
                "grad_norm": [round(v, 6) for v in grad_norm.tolist()],
                "update_norm": [round(v, 6) for v in update_norm.tolist()],
                "weight_norm": [round(v, 5) for v in w_norm.tolist()],
                "drift_from_init": [round(v, 5) for v in drift.tolist()],
                "router_row_norm": [round(v, 5) for v in router_norm.tolist()],
                "mean_router_logit": [round(v, 5) for v in mlogit.tolist()],
                "gini": gini(c),
                "dead_fraction": float((c == 0).float().mean().item()),
            })

    def _mean(xs):
        xs = [x for x in xs if x == x]   # drop NaN
        return sum(xs) / len(xs) if xs else float("nan")

    out = {"load/corr_share_vs_gradnorm": _mean(corr_grad)}
    if corr_update:
        out["load/corr_share_vs_update"] = _mean(corr_update)
    if corr_update_alive:
        out["load/corr_share_vs_update_alive"] = _mean(corr_update_alive)
    return {k: v for k, v in out.items() if v == v}


def train(args):
    # --- DDP ---
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        dist.init_process_group("nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{ddp_local_rank}")
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_local_rank, world_size, master_process = 0, 1, True
        device = torch.device(args.device) if args.device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if not master_process:
        os.environ["WANDB_MODE"] = "disabled"

    tcfg = dict(TRAIN_CONFIG)
    for key in ("n_steps", "batch_size", "grad_accum", "seq_len"):
        if getattr(args, key) is not None:
            tcfg[key] = getattr(args, key)
    if args.dry_run:
        tcfg.update(n_steps=100, eval_every=50, log_every=10)
        if master_process:
            print("DRY RUN: 100 steps")

    use_amp = device.type == "cuda"
    amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

    arch_name, config_fn = ARCH_REGISTRY[args.arch]
    run_name = build_run_name(args)
    save_dir = Path(args.save_dir)
    if master_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*70}\n  {arch_name}  |  run: {run_name}\n{'='*70}")
        print(f"Device: {device} | DDP world size: {world_size}")
        if device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name()} "
                  f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # --- Model ---
    config = config_fn(scale=args.scale)
    config._experts_implementation = "grouped_mm"
    if master_process:
        print(f"\nBuilding model (scale={args.scale}) ...")

    if args.arch == "qwen3moe":
        from transformers import Qwen3MoeForCausalLM
        model = Qwen3MoeForCausalLM(config)
    else:
        from transformers import OlmoeForCausalLM
        model = OlmoeForCausalLM(config)

    install_router_loss_patch(args.aux_loss_coef, args.z_loss_coef, master_process)
    reinit_weights_depth_scaled(model, config, master_process)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    model = model.to(device)
    if is_ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False,
                    broadcast_buffers=False, gradient_as_bucket_view=True)

    total_params = sum(p.numel() for p in model.parameters())
    if master_process:
        print(f"  Params: {total_params:,} ({total_params/1e9:.2f}B)")
        verify_grouped_mm(model, config, master_process)

    # --- Optimizer ---
    opt_lr = args.lr if args.lr is not None else tcfg["lr"]
    if master_process:
        print(f"\nOptimizer: {args.optimizer} @ lr={opt_lr}")
    optimizer, expert_lr_groups = build_optimizer(model, args.optimizer, opt_lr, master_process)

    per_expert_lr_active = (args.per_expert_lr_alpha != 0.0) or args.freeze_dead_experts
    if per_expert_lr_active and master_process:
        print(f"  Per-expert LR: alpha={args.per_expert_lr_alpha}, "
              f"freeze_dead={args.freeze_dead_experts}, "
              f"normalize={args.per_expert_lr_normalize}, "
              f"clamp_max={args.per_expert_lr_clamp_max}")

    # --- Resume ---
    start_step, best_eval_loss, wandb_run_id = 0, float("inf"), None
    if args.auto_resume and not args.resume:
        ckpts = sorted(save_dir.glob(f"{run_name}_step*.pt"),
                       key=lambda p: int(p.stem.rsplit("step", 1)[1]))
        if ckpts:
            args.resume = str(ckpts[-1])
            if master_process:
                print(f"\n  [auto-resume] latest: {args.resume}")
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw = model.module if is_ddp else model
        raw.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            opts = ([optimizer.muon, optimizer.adamw]
                    if isinstance(optimizer, DualOptimizer) else [optimizer])
            for opt in opts:
                for state in opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
        start_step = ckpt.get("step", -1) + 1
        best_eval_loss = ckpt.get("best_eval_loss") or float("inf")
        wandb_run_id = ckpt.get("wandb_run_id")
        if master_process:
            print(f"  Resumed at step {start_step} (best {best_eval_loss:.4f})")
    elif args.resume and master_process:
        print(f"  Checkpoint {args.resume} not found; starting fresh.")

    # --- Data ---
    train_loader = get_dataloader(seq_len=tcfg["seq_len"], batch_size=tcfg["batch_size"],
                                 num_workers=4)
    eval_batches = get_eval_batches(seq_len=tcfg["seq_len"], batch_size=tcfg["batch_size"],
                                    n_batches=tcfg["eval_batches"])
    grad_accum = tcfg["grad_accum"]
    eff_batch_tokens = tcfg["batch_size"] * grad_accum * tcfg["seq_len"] * world_size
    top_k = config.num_experts_per_tok
    if master_process:
        print(f"\nData: {tcfg['batch_size']}x{grad_accum}x{tcfg['seq_len']}x{world_size}gpu "
              f"= {eff_batch_tokens:,} tok/step")
        print(f"  Expected router assignments per layer per step: "
              f"{eff_batch_tokens * top_k:,} (= tokens x top_k={top_k})")

    # --- Instrumentation ---
    tracker = ExpertLoadTracker(model, top_k=top_k, ema=args.expert_count_ema) \
        if args.expert_tracking else None
    if tracker is not None and master_process:
        print(f"  Expert load tracking: {tracker.n_layers} routers hooked")

    expert_tensors = {}
    for li, layer in enumerate(find_layers(model)):
        try:
            expert_tensors[li] = {
                "gate_up": layer.mlp.experts.gate_up_proj,
                "down": layer.mlp.experts.down_proj,
                "router": layer.mlp.gate.weight,
            }
        except AttributeError:
            continue   # dense layer in an otherwise-MoE stack

    init_weights = {}
    if args.track_expert_drift:
        for li, t in expert_tensors.items():
            init_weights[li] = {k: t[k].detach().clone().to(torch.bfloat16)
                                for k in ("gate_up", "down")}
        if master_process:
            print("  Tracking ||w - w_init|| per expert (bf16 copy on device)")

    stats_writer = None
    if master_process and args.expert_stats_full_every:
        stats_writer = ExpertStatsWriter(save_dir / f"{run_name}_expert_stats.jsonl")
        print(f"  Per-expert records: {stats_writer.path}")

    # Per-micro-batch, per-rank counts: input for the expert-parallel simulator
    # (per-rank max/mean under different placements) and the micro-batch vs
    # global-batch comparison. One record holds [ranks, micro-batches, layers, experts].
    mb_writer = None
    if master_process and args.microbatch_stats_every and tracker is not None:
        mb_writer = ExpertStatsWriter(save_dir / f"{run_name}_microbatch_counts.jsonl")
        print(f"  Micro-batch counts: {mb_writer.path}")

    # --- wandb ---
    # Run identity (group, tags, notes, git state, all CLI flags) is handled in
    # run_tracking.py; only the experiment-specific config is assembled here.
    run = None
    if HAS_WANDB and not args.no_wandb and master_process:
        run = init_wandb(args, run_name, n_steps=tcfg["n_steps"],
                         resume_id=wandb_run_id if args.resume else None,
                         config={
            "arch": args.arch, "scale": args.scale, "optimizer": args.optimizer,
            "lr": opt_lr, "aux_loss_coef": args.aux_loss_coef,
            "z_loss_coef": args.z_loss_coef,
            "per_expert_lr_alpha": args.per_expert_lr_alpha,
            "freeze_dead_experts": args.freeze_dead_experts,
            "per_expert_lr_normalize": args.per_expert_lr_normalize,
            "expert_count_ema": args.expert_count_ema,
            "num_experts": config.num_experts, "num_experts_per_tok": top_k,
            "num_hidden_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "total_params": total_params,
            "batch_size": tcfg["batch_size"], "grad_accum": grad_accum,
            "seq_len": tcfg["seq_len"], "ddp_world_size": world_size,
            "eff_batch_tokens": eff_batch_tokens,
        })

    # --- Loop ---
    n_steps = tcfg["n_steps"]
    step = start_step
    losses = []
    t_start = time.time()
    model.train()

    train_iter = iter(train_loader)
    if start_step > 0:
        # The stream is not seekable, so resuming means replaying it micro-batch
        # by micro-batch. Costly; prefer restarting short runs from scratch.
        to_skip = start_step * grad_accum
        if master_process:
            print(f"  Fast-forwarding {to_skip} micro-batches ...", flush=True)
        for _ in range(to_skip):
            try:
                next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                next(train_iter)

    pbar = tqdm(total=n_steps, initial=start_step, desc=run_name, ncols=100) \
        if (HAS_TQDM and master_process) else None

    while step < n_steps:
        # LR schedule. Muon rides the full rate; AdamW rides a parallel 1/10 one.
        min_lr = opt_lr * (tcfg["min_lr"] / tcfg["lr"])
        lr = get_lr(step, tcfg["warmup_steps"], n_steps, opt_lr, min_lr)
        adamw_lr = get_lr(step, tcfg["warmup_steps"], n_steps, opt_lr * 0.1, min_lr * 0.1)
        if isinstance(optimizer, DualOptimizer):
            for pg in optimizer.muon.param_groups:
                pg["lr"] = lr
            for pg in optimizer.adamw.param_groups:
                pg["lr"] = adamw_lr
        else:
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # --- Gradient accumulation ---
        optimizer.zero_grad(set_to_none=True)
        if tracker is not None:
            tracker.reset()
        accum_loss = accum_aux = accum_lm_loss = 0.0

        for micro_step in range(grad_accum):
            try:
                input_ids, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_ids, targets = next(train_iter)
            input_ids = input_ids.to(device)

            last_micro = micro_step == grad_accum - 1
            sync_ctx = model.no_sync() if (is_ddp and not last_micro) else nullcontext()

            with sync_ctx:
                with amp_ctx():
                    # Count during the forward only: with non-reentrant gradient
                    # checkpointing the forward is replayed inside backward, and
                    # the router hooks would fire a second time.
                    if tracker is not None:
                        tracker.arm()
                    # targets from data.py is already shifted, so labels=input_ids
                    # avoids a second shift inside HF.
                    out = model(input_ids=input_ids, labels=input_ids,
                                output_router_logits=True)
                    if tracker is not None:
                        tracker.disarm()
                    aux_loss = getattr(out, "aux_loss", None)
                    loss_scaled = out.loss / grad_accum
                loss_scaled.backward()

            accum_loss += out.loss.item() / grad_accum
            if aux_loss is not None and not isinstance(aux_loss, int):
                accum_aux += aux_loss.item() / grad_accum
                # out.loss = lm_loss + AUX_CARRIER * out.aux_loss, so subtracting
                # that term recovers the pure cross-entropy. This is the only
                # loss comparable across runs with different aux coefficients.
                accum_lm_loss += (out.loss.item() - AUX_CARRIER * aux_loss.item()) / grad_accum
            else:
                accum_lm_loss += out.loss.item() / grad_accum

        if is_ddp:
            m = torch.tensor([accum_loss, accum_aux, accum_lm_loss], device=device)
            dist.all_reduce(m, op=dist.ReduceOp.AVG)
            accum_loss, accum_aux, accum_lm_loss = m.tolist()

        # --- Per-expert counts and learning rates ---
        # all_reduce BEFORE deriving any learning rate: every rank must compute
        # identical per-expert LRs or the replicas silently diverge.
        load_metrics, raw_counts, lr_counts = {}, {}, {}
        if tracker is not None:
            tracker.synchronize()
            raw_counts = tracker.counts()
            lr_counts = tracker.update_ema() if args.expert_count_ema > 0 else raw_counts
            if master_process and (step % args.expert_stats_every == 0
                                   or step < args.expert_stats_dense_until):
                load_metrics = tracker.layer_metrics()
                load_metrics.update(tracker.microbatch_metrics())
            # Rank-independent condition: microbatch_counts(gather=True) is a
            # collective, so every rank must take this branch on the same steps.
            if args.microbatch_stats_every and (step % args.microbatch_stats_every == 0
                                                or step < args.microbatch_stats_dense_until):
                mbc = tracker.microbatch_counts(gather=True)
                if mb_writer is not None and mbc is not None:
                    mb_writer.write({
                        "step": step,
                        "ranks": mbc.size(0), "micro_batches": mbc.size(1),
                        "layers": tracker.layer_ids(), "top_k": top_k,
                        "tokens_per_micro_batch": tcfg["batch_size"] * tcfg["seq_len"],
                        "counts": mbc.tolist(),   # [ranks][micro_batches][layers][experts]
                    })

        if per_expert_lr_active and lr_counts:
            # The schedule above reset every group to the base rate, so these
            # multipliers do not compound across steps.
            for li, counts_1d in lr_counts.items():
                scales = compute_lr_scales(
                    counts_1d,
                    alpha=args.per_expert_lr_alpha,
                    freeze_dead=args.freeze_dead_experts,
                    normalize=args.per_expert_lr_normalize,
                    clamp_max=args.per_expert_lr_clamp_max,
                    dead_counts=raw_counts.get(li),   # deadness is per-step, not EMA
                )
                for ei in range(scales.numel()):
                    for pg in expert_lr_groups.get((li, ei), ()):
                        pg["lr"] *= float(scales[ei].item())

        # Snapshot expert weights so ||Δw|| per expert is exact.
        do_full_stats = (stats_writer is not None and args.expert_stats_full_every
                         and step % args.expert_stats_full_every == 0)
        w_before = {}
        if do_full_stats:
            for li, t in expert_tensors.items():
                w_before[li] = {k: t[k].detach().clone() for k in ("gate_up", "down")}

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        optimizer.step()

        if tracker is not None and (do_full_stats or load_metrics):
            with torch.no_grad():
                load_metrics.update(expert_drift_stats(
                    step, tracker, expert_tensors, w_before, init_weights,
                    writer=stats_writer if do_full_stats else None))
        w_before = {}

        if master_process and run is not None and load_metrics:
            # Own cadence: dense early, where the imbalance is largest.
            wandb.log(load_metrics, step=step)

        losses.append(accum_loss)

        # --- Logging ---
        if master_process and (step % tcfg["log_every"] == 0 or step < 10):
            window = losses[-tcfg["log_every"]:]
            avg_loss = sum(window) / len(window)
            elapsed = time.time() - t_start
            tps = ((step - start_step) + 1) * eff_batch_tokens / elapsed
            load_str = ""
            if load_metrics:
                load_str = (f" gini={load_metrics.get('load/gini', float('nan')):.3f}"
                            f" dead={load_metrics.get('load/dead_expert_fraction', float('nan')):.3f}")
            if pbar:
                pbar.set_postfix(loss=f"{avg_loss:.4f}", tps=f"{tps:.0f}")
            print(f"  [Step {step}] loss={avg_loss:.4f} lm={accum_lm_loss:.4f} "
                  f"aux={accum_aux:.4f} tps={tps:.0f} lr={lr:.5f}{load_str} "
                  f"elapsed={elapsed:.0f}s", flush=True)

            if run:
                log = {
                    "train/loss": accum_loss,
                    "train/lm_loss": accum_lm_loss,   # compare runs on THIS
                    "train/aux_loss": accum_aux,
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "train/tokens_per_sec": tps,
                    "train/tokens_seen": (step + 1) * eff_batch_tokens,
                    "train/lr_muon": lr,
                    "train/lr_adamw": adamw_lr,
                }
                if device.type == "cuda":
                    log["system/gpu_mem_gb"] = torch.cuda.memory_allocated() / 1e9
                    log["system/gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
                wandb.log(log, step=step)

        # --- Eval + checkpoints ---
        if step > 0 and step % tcfg["eval_every"] == 0:
            eval_loss, eval_ppl = evaluate(model, eval_batches, device)
            if master_process:
                print(f"\n  [Step {step}] eval loss={eval_loss:.4f} ppl={eval_ppl:.2f}")
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    save_checkpoint(save_dir / f"{run_name}_best.pt", model, optimizer, step,
                                    best_eval_loss, args, config, run, is_ddp)
                if run:
                    wandb.log({"eval/loss": eval_loss, "eval/perplexity": eval_ppl,
                               "eval/best_loss": best_eval_loss}, step=step)

        if step > 0 and step % tcfg["save_every"] == 0 and master_process:
            save_checkpoint(save_dir / f"{run_name}_step{step}.pt", model, optimizer, step,
                            best_eval_loss, args, config, run, is_ddp)

        if pbar:
            pbar.update(1)
        step += 1

    if pbar:
        pbar.close()
    if tracker is not None:
        tracker.remove()
    if stats_writer is not None:
        stats_writer.close()
    if mb_writer is not None:
        mb_writer.close()

    # --- Final ---
    final_loss, final_ppl = evaluate(model, eval_batches, device)
    elapsed = time.time() - t_start
    results = {
        "run_name": run_name, "arch": args.arch, "scale": args.scale,
        "optimizer": args.optimizer, "aux_loss_coef": args.aux_loss_coef,
        "per_expert_lr_alpha": args.per_expert_lr_alpha,
        "freeze_dead_experts": args.freeze_dead_experts,
        "total_params": total_params,
        "final_eval_loss": final_loss, "final_eval_ppl": final_ppl,
        "best_eval_loss": best_eval_loss,
        "training_time_sec": elapsed,
        "tokens_per_sec": (n_steps - start_step) * eff_batch_tokens / elapsed,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0,
    }
    if master_process:
        save_checkpoint(save_dir / f"{run_name}_step{step}.pt", model, optimizer, step,
                        best_eval_loss, args, config, run, is_ddp)
        print(f"\n{'='*70}\n  Final: loss={final_loss:.4f} ppl={final_ppl:.2f} "
              f"best={best_eval_loss:.4f} time={elapsed:.0f}s\n{'='*70}")
        if run:
            wandb.log({f"final/{k}": v for k, v in results.items()
                       if isinstance(v, (int, float))})
            wandb.finish()
        with open(save_dir / f"{run_name}_results.json", "w") as f:
            json.dump(results, f, indent=2)

    if is_ddp:
        dist.destroy_process_group()
    return results


def save_checkpoint(path, model, optimizer, step, best_eval_loss, args, config, run, is_ddp):
    """Keys match moe_train.py's, so eval_downstream.py --tie-group-size 1 reads these."""
    raw = model.module if is_ddp else model
    raw = raw._orig_mod if hasattr(raw, "_orig_mod") else raw
    torch.save({
        "step": step,
        "model_state_dict": raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_eval_loss": best_eval_loss,
        "arch": args.arch,
        "config": config.to_dict(),
        "expand_tied_experts": None,   # present for eval_downstream compatibility
        "wandb_run_id": run.id if run else None,
    }, path)
    print(f"  → checkpoint: {path}")


# ──────────────────────────────────────────────────────────────────────
# 7. CLI
# ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="conf", config_name="imbalance")
def main(cfg: DictConfig):
    """Entry point. Config: conf/imbalance.yaml, overridable as key=value on the CLI.

    The composed config is turned into a plain Namespace so the rest of the file
    keeps using args.<name> exactly as with the former argparse interface.
    """
    args = argparse.Namespace(**OmegaConf.to_container(cfg, resolve=True))

    if args.arch not in ARCH_REGISTRY:
        raise SystemExit(f"arch must be one of {list(ARCH_REGISTRY)}, got {args.arch!r}")
    if args.scale not in ("regular", "small", "tiny"):
        raise SystemExit(f"scale must be regular | small | tiny, got {args.scale!r}")
    if args.optimizer not in ("adamw", "muon"):
        raise SystemExit(f"optimizer must be adamw | muon, got {args.optimizer!r}")

    # Keep a copy of the exact config next to the checkpoints, named like them.
    if int(os.environ.get("RANK", "0")) == 0:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, Path(args.save_dir) / f"{build_run_name(args)}_config.yaml")

    if (args.per_expert_lr_alpha != 0.0 or args.freeze_dead_experts):
        if args.optimizer != "muon":
            raise SystemExit(
                "per_expert_lr_alpha / freeze_dead_experts require optimizer=muon: "
                "under AdamW the experts stay as single 3D tensors, so there is no "
                "per-expert parameter to attach a learning rate to.")
        if not args.expert_tracking:
            raise SystemExit("Per-expert LR needs the token counts; set expert_tracking=true.")
    if args.aux_loss_coef < 0 or args.z_loss_coef < 0:
        raise SystemExit("Loss coefficients must be >= 0.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train(args)


if __name__ == "__main__":
    main()
