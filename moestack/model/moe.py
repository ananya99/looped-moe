"""The MoE layer, split into the three pieces that research ideas actually change.

    Router      logits -> top-k experts + gate weights        (routing / balancing ideas)
    Dispatcher  tokens -> grouped by expert -> back            (expert parallelism lives here)
    Experts     grouped SwiGLU over [E_local, ...] weights     (kernels, precision)

`MoE.forward` returns a `RouterOut` next to the hidden states. Losses, load
statistics and any future per-expert policy read routing information from that
return value -- never from forward hooks -- so activation checkpointing (which
replays the forward inside backward) cannot double-count anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelArgs


# ──────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RouterOut:
    logits: torch.Tensor        # [N, E] raw router logits (autocast dtype), carries grad
    topk_idx: torch.Tensor      # [N, k] int64
    topk_weight: torch.Tensor   # [N, k] gate weights (logits dtype), carries grad
    counts: torch.Tensor        # [E] int64, tokens assigned to each expert (no grad)


class Router(nn.Module):
    """Softmax-then-top-k token-choice router (OLMoE / Qwen3-MoE convention)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_experts, self.top_k = args.n_experts, args.top_k
        self.norm_topk_prob = args.norm_topk_prob
        self.weight = nn.Parameter(torch.empty(args.n_experts, args.dim))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> RouterOut:
        logits = F.linear(x, self.weight)                       # [N, E]
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        topk_w, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        with torch.no_grad():
            counts = torch.bincount(topk_idx.flatten(), minlength=self.n_experts)
        return RouterOut(logits, topk_idx, topk_w.to(logits.dtype), counts)

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)


# ──────────────────────────────────────────────────────────────────────
# Dispatcher: the expert-parallel seam
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DispatchHandle:
    """Whatever `combine` needs to undo `dispatch`."""
    token_idx: torch.Tensor     # [S] source token of each sorted row
    weight: torch.Tensor        # [S] gate weight of each sorted row
    num_tokens: int


class Dispatcher(Protocol):
    def dispatch(self, x: torch.Tensor, r: RouterOut
                 ) -> tuple[torch.Tensor, torch.Tensor, DispatchHandle]:
        """x: [N, D] local tokens. Returns (rows grouped by *local* expert [S, D],
        tokens_per_local_expert [E_local] int64, handle)."""
        ...

    def combine(self, y: torch.Tensor, h: DispatchHandle) -> torch.Tensor:
        """y: [S, D] expert outputs in dispatch order. Returns [N, D]."""
        ...


class LocalDispatcher:
    """EP = 1: every expert is on this rank, so dispatch is a sort and combine is a
    weighted scatter-add. An all-to-all dispatcher implements the same two methods
    with communication in between; nothing else in the model changes."""

    def __init__(self, n_experts: int):
        self.n_experts = n_experts

    def dispatch(self, x, r):
        k = r.topk_idx.size(-1)
        flat_idx = r.topk_idx.reshape(-1)                       # [N*k]
        sorted_expert, perm = torch.sort(flat_idx, stable=True)
        token_idx = perm // k
        tokens_per_expert = torch.bincount(sorted_expert, minlength=self.n_experts)
        handle = DispatchHandle(token_idx=token_idx,
                                weight=r.topk_weight.reshape(-1)[perm],
                                num_tokens=x.size(0))
        return x[token_idx], tokens_per_expert, handle

    def combine(self, y, h):
        y = y * h.weight.unsqueeze(-1).to(y.dtype)
        out = y.new_zeros(h.num_tokens, y.size(-1))
        return out.index_add_(0, h.token_idx, y)


# ──────────────────────────────────────────────────────────────────────
# Experts
# ──────────────────────────────────────────────────────────────────────

def _grouped_mm_available() -> bool:
    return hasattr(F, "grouped_mm")


class GroupedExperts(nn.Module):
    """E SwiGLU experts stored as 3D tensors, HF layout:
        gate_up_proj [E, 2*I, D]   (gate rows first, then up rows)
        down_proj    [E, D, I]
    Sharding these along dim 0 only (one whole matrix per expert per rank) is the
    rule that keeps Muon's Newton-Schulz and any per-expert logic rank-local.
    """

    def __init__(self, args: ModelArgs, n_local_experts: int | None = None):
        super().__init__()
        E = n_local_experts or args.n_experts
        self.n_experts, self.inter = E, args.moe_inter_dim
        self.impl = args.experts_impl
        self.gate_up_proj = nn.Parameter(torch.empty(E, 2 * args.moe_inter_dim, args.dim))
        self.down_proj = nn.Parameter(torch.empty(E, args.dim, args.moe_inter_dim))
        self.reset_parameters()

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        if self.impl == "grouped_mm" and _grouped_mm_available():
            return self._forward_grouped(x, tokens_per_expert)
        return self._forward_loop(x, tokens_per_expert)

    # Inductor only lowers grouped_mm for bf16 operands; the original trainer ran the
    # experts in fp32 (weights' dtype). Keep that numerics under torch.compile by
    # leaving this call eager (one graph break per block). Running experts in bf16
    # under compile is a deliberate future speed/precision trade-off, not a default.
    @torch.compiler.disable
    def _forward_grouped(self, x, tokens_per_expert):
        offs = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
        # grouped_mm is not autocast-aware: run it in the weights' dtype, exactly as
        # the HF grouped_mm path used by the original trainer does.
        x = x.to(self.gate_up_proj.dtype)
        h = F.grouped_mm(x, self.gate_up_proj.transpose(-2, -1), offs=offs)
        gate, up = h.chunk(2, dim=-1)
        h = F.silu(gate) * up
        return F.grouped_mm(h, self.down_proj.transpose(-2, -1), offs=offs)

    def _forward_loop(self, x, tokens_per_expert):
        """Reference implementation: one matmul per expert. Slow, obviously correct."""
        outs, start = [], 0
        for e, n in enumerate(tokens_per_expert.tolist()):
            xe = x[start:start + n]
            gate, up = F.linear(xe, self.gate_up_proj[e]).chunk(2, dim=-1)
            outs.append(F.linear(F.silu(gate) * up, self.down_proj[e]))
            start += n
        return torch.cat(outs, dim=0) if outs else x.new_zeros(0, self.down_proj.size(1))

    def reset_parameters(self):
        nn.init.normal_(self.gate_up_proj, std=0.02)
        nn.init.normal_(self.down_proj, std=0.02)


# ──────────────────────────────────────────────────────────────────────
# MoE layer
# ──────────────────────────────────────────────────────────────────────

class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.router = Router(args)
        self.experts = GroupedExperts(args)
        # Not a submodule: swapping it (e.g. for EP) must not change the state dict.
        self.dispatcher: Dispatcher = LocalDispatcher(args.n_experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, RouterOut]:
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        r = self.router(flat)
        rows, tokens_per_expert, handle = self.dispatcher.dispatch(flat, r)
        y = self.experts(rows, tokens_per_expert)
        out = self.dispatcher.combine(y, handle)
        return out.to(x.dtype).view(B, T, D), r
