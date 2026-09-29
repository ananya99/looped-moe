"""
expert_stats.py — Per-expert load measurement and per-expert learning-rate rules
===============================================================================

Instrumentation for the expert-imbalance experiment: how does the number of
tokens routed to an expert influence the learning of that expert's weights and
of its router row, and does varying the learning rate per expert change it?

Two pieces:

  ExpertLoadTracker
      Forward hooks on every ``mlp.gate`` router that accumulate the exact
      top-k token counts per expert, per layer, over *all* micro-steps of a
      gradient-accumulation window and *all* DDP ranks. This is the global-batch
      assignment, which is the quantity the experiment is about — not the
      last-micro-batch, rank-0, top-1 approximation that the existing
      ``router/*`` dashboard metrics in moe_train.py report.

  compute_lr_scales
      Maps per-expert token shares to per-expert learning-rate multipliers,
      implementing the ``lr_e ∝ s_e^alpha`` family plus the freeze-dead-experts
      control.

Why counts must be exact
------------------------
Muon's Newton-Schulz step normalises the update to unit spectral norm, so the
*magnitude* of an expert's step is independent of how many tokens it saw; only
the direction and its noise level differ. The interesting quantities are
therefore the joint distribution of (token share, weight drift) and the
behaviour of zero-token experts, both of which are destroyed by approximate
counting.

Gradient checkpointing
----------------------
With ``use_reentrant=False`` the forward is recomputed during backward, and
module forward hooks fire again on the recomputed pass. The tracker therefore
counts only while explicitly armed: the training loop arms it immediately
before the forward and disarms it immediately after, so recomputation during
backward is ignored. Without this, every count would be doubled.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.distributed as dist


def find_layers(model):
    """Return the decoder layer list of an HF causal-LM, unwrapping DDP/compile."""
    raw = model
    if hasattr(raw, "module"):          # DDP
        raw = raw.module
    if hasattr(raw, "_orig_mod"):       # torch.compile
        raw = raw._orig_mod
    if hasattr(raw, "model") and hasattr(raw.model, "layers"):
        return raw.model.layers
    if hasattr(raw, "layers"):
        return raw.layers
    raise ValueError("Cannot locate decoder layer list on model")


class ExpertLoadTracker:
    """Accumulate exact per-layer, per-expert top-k token counts over a step.

    Usage per optimizer step::

        tracker.reset()
        for micro_step in range(grad_accum):
            tracker.arm()
            out = model(...)
            tracker.disarm()
            loss.backward()
        tracker.synchronize()      # all-reduce across DDP ranks
        counts = tracker.counts()  # dict: layer_idx -> LongTensor[E]

    ``synchronize`` must be called before any per-expert learning rate is
    derived from the counts: every rank has to compute identical learning rates
    or the replicas silently diverge.
    """

    def __init__(self, model, top_k: int, ema: float = 0.9):
        self.top_k = top_k
        self.ema = ema
        self._armed = False
        self._counts = {}       # layer_idx -> LongTensor[E]
        self._ema_counts = {}   # layer_idx -> FloatTensor[E]
        self._logit_sums = {}   # layer_idx -> FloatTensor[E], summed over tokens
        self._token_totals = {} # layer_idx -> int, tokens seen (for the mean)
        self._handles = []
        self._n_experts = {}    # layer_idx -> E

        layers = find_layers(model)
        self.n_layers = 0
        for li, layer in enumerate(layers):
            # Architectures can interleave dense layers among the MoE ones
            # (Qwen3-MoE's decoder_sparse_step); skip anything without a router.
            gate = getattr(getattr(layer, "mlp", None), "gate", None)
            if gate is None:
                continue
            self._handles.append(gate.register_forward_hook(self._make_hook(li)))
            self.n_layers += 1

    # -- hook plumbing -------------------------------------------------

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if not self._armed:
                return
            indices, n_experts = self._extract_indices(module, output)
            if indices is None:
                return
            counts = torch.bincount(indices.flatten(), minlength=n_experts)
            prev = self._counts.get(layer_idx)
            self._counts[layer_idx] = counts if prev is None else prev + counts
            self._n_experts[layer_idx] = n_experts

            # Mean router logit per expert, for separating "the router stopped
            # proposing this expert" from "the router proposes it but top-k never
            # reaches it".
            logits = self._extract_logits(output)
            if logits is not None:
                flat = logits.reshape(-1, n_experts).float()
                lsum = flat.sum(dim=0)
                prev_l = self._logit_sums.get(layer_idx)
                self._logit_sums[layer_idx] = lsum if prev_l is None else prev_l + lsum
                self._token_totals[layer_idx] = self._token_totals.get(layer_idx, 0) + flat.size(0)
        return hook

    @staticmethod
    def _extract_logits(output):
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        if isinstance(output, torch.Tensor):
            return output
        return None

    def _extract_indices(self, module, output):
        """Get the [T, k] top-k expert indices from a router module's output.

        HF 5.x ``OlmoeTopKRouter`` / ``Qwen3MoeTopKRouter`` return
        ``(router_logits, router_scores, router_indices)``; the sparse-MoE block
        discards the logits. Taking the indices straight off the router avoids
        recomputing top-k and is exact. Older or differently-shaped routers that
        return bare logits are handled by falling back to an explicit top-k.
        """
        if isinstance(output, (tuple, list)):
            if len(output) >= 3 and isinstance(output[2], torch.Tensor) and output[2].dtype in (torch.int32, torch.int64):
                idx = output[2]
                n_experts = getattr(module, "num_experts", None) or int(idx.max().item()) + 1
                return idx, n_experts
            logits = output[0]
        elif isinstance(output, torch.Tensor):
            logits = output
        else:
            return None, 0

        if not isinstance(logits, torch.Tensor) or logits.dim() < 2:
            return None, 0
        n_experts = logits.size(-1)
        flat = logits.reshape(-1, n_experts)
        k = min(self.top_k, n_experts)
        return torch.topk(flat, k, dim=-1).indices, n_experts

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # -- step lifecycle ------------------------------------------------

    def arm(self):
        self._armed = True

    def disarm(self):
        self._armed = False

    def reset(self):
        self._counts = {}
        self._logit_sums = {}
        self._token_totals = {}

    def synchronize(self):
        """Sum counts across DDP ranks so every rank sees the global batch."""
        if not (dist.is_available() and dist.is_initialized()):
            return
        for li in sorted(self._counts):
            c = self._counts[li]
            dist.all_reduce(c, op=dist.ReduceOp.SUM)
            self._counts[li] = c
        for li in sorted(self._logit_sums):
            l = self._logit_sums[li]
            dist.all_reduce(l, op=dist.ReduceOp.SUM)
            self._logit_sums[li] = l
            self._token_totals[li] = self._token_totals.get(li, 0) * dist.get_world_size()

    def counts(self):
        return self._counts

    def update_ema(self):
        """Fold this step's counts into the EMA used for smoothed LR rules.

        Instantaneous per-step counts are very noisy where n_e is small, which
        is exactly the regime the experiment cares about.
        """
        for li, c in self._counts.items():
            c = c.float()
            prev = self._ema_counts.get(li)
            self._ema_counts[li] = c if prev is None else self.ema * prev + (1.0 - self.ema) * c
        return self._ema_counts

    def ema_counts(self):
        return self._ema_counts

    def mean_logits(self):
        """layer_idx -> FloatTensor[E] of mean router logit over this step's tokens."""
        out = {}
        for li, lsum in self._logit_sums.items():
            n = max(1, self._token_totals.get(li, 1))
            out[li] = lsum / n
        return out

    # -- derived quantities --------------------------------------------

    @staticmethod
    def shares(counts_1d: torch.Tensor) -> torch.Tensor:
        """Token share s_e = n_e / mean(n): 1.0 is a perfectly balanced expert."""
        c = counts_1d.float()
        mean = c.mean()
        if mean <= 0:
            return torch.ones_like(c)
        return c / mean

    def layer_metrics(self) -> dict:
        """Aggregate imbalance metrics over layers, from exact global counts."""
        if not self._counts:
            return {}
        ginis, entropies, deads, maxshares = [], [], [], []
        for li in sorted(self._counts):
            c = self._counts[li].float()
            total = c.sum()
            if total <= 0:
                continue
            p = c / total
            E = c.numel()
            ginis.append(gini(c))
            nz = p[p > 0]
            entropies.append(float(-(nz * nz.log()).sum().item() / math.log(E)) if E > 1 else 0.0)
            deads.append(float((c == 0).sum().item()) / E)
            maxshares.append(float(p.max().item()))
        if not ginis:
            return {}
        n = len(ginis)
        return {
            "load/gini": sum(ginis) / n,
            "load/normalized_entropy": sum(entropies) / n,
            "load/dead_expert_fraction": sum(deads) / n,
            "load/max_expert_fraction": sum(maxshares) / n,
            "load/gini_max_layer": max(ginis),
            "load/dead_fraction_max_layer": max(deads),
        }


def gini(counts_1d: torch.Tensor) -> float:
    """Gini coefficient of a non-negative count vector. 0 = uniform, →1 = collapsed."""
    c = counts_1d.float().flatten()
    total = c.sum()
    if total <= 0:
        return 0.0
    srt, _ = torch.sort(c)
    n = srt.numel()
    idx = torch.arange(1, n + 1, device=srt.device, dtype=srt.dtype)
    return float(((2 * idx - n - 1) * srt).sum().item() / (n * total.item()))


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rank correlation between two 1-D tensors.

    Returns NaN when either input has no rank variance — which happens often
    here (all experts dead, or every expert taking an identically-sized Muon
    step), and must not be reported as a correlation of zero.
    """
    x = x.float().flatten()
    y = y.float().flatten()
    if x.numel() < 3 or x.numel() != y.numel():
        return float("nan")
    rx = _rank(x)
    ry = _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    if denom <= 0:
        return float("nan")
    return float((rx @ ry / denom).item())


def _rank(v: torch.Tensor) -> torch.Tensor:
    """Ranks with ties averaged (the 'average' method).

    Ties are the common case in this experiment: many experts sit at exactly
    zero tokens, and Newton-Schulz makes many update norms near-identical.
    Breaking ties arbitrarily would invent an ordering and report correlation
    where there is none.
    """
    n = v.numel()
    order = torch.argsort(v)
    sorted_v = v[order]
    ranks = torch.empty(n, dtype=torch.float32, device=v.device)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def compute_lr_scales(
    counts_1d: torch.Tensor,
    alpha: float = 0.0,
    freeze_dead: bool = False,
    normalize: bool = True,
    clamp_min: float = 1e-3,
    clamp_max: float = 10.0,
    dead_counts: torch.Tensor = None,
) -> torch.Tensor:
    """Per-expert learning-rate multipliers from this step's token counts.

    ``lr_e = clamp(s_e ** alpha)`` with ``s_e = n_e / mean(n)`` the token share.

      alpha =  0    uniform (the baseline; multipliers are all 1.0)
      alpha =  1    token-proportional, restoring roughly the SGD-like scaling
                    that Newton-Schulz normalisation removes
      alpha =  0.5  square-root compromise
      alpha = -0.5  the opposite sign: compensate cold experts *upward*

    ``freeze_dead`` sets the multiplier of zero-token experts to exactly 0.
    Because both Muon and AdamW scale decoupled weight decay by the learning
    rate (``param.mul_(1 - lr * weight_decay)``), a multiplier of 0 suppresses
    the weight-decay shrinkage and the stale-momentum step together, which is
    the intended control.

    Experts with zero tokens would otherwise get ``0 ** negative = inf``, so the
    dead case is resolved before exponentiation.

    With ``normalize`` the multipliers are rescaled to average 1.0 over the
    experts that received tokens. Without it, alpha also changes the *average*
    learning rate across the layer, so a difference between alpha=0 and alpha=1
    could not be attributed to the redistribution rather than to an overall
    rate change. Normalising keeps the total step budget fixed and makes the
    alpha sweep a controlled comparison.

    Clamping is deliberately asymmetric: a floor far below 1 only means an
    expert learns slowly (safe), whereas a high ceiling multiplies the Muon
    learning rate on a hot expert and can destabilise training.

    ``dead_counts`` supplies the counts used to decide which experts are dead,
    separately from the counts driving the share. This matters when
    ``counts_1d`` is an EMA: an expert that received nothing *this step* still
    carries EMA mass from earlier steps, so freezing has to be keyed on the raw
    per-step counts or it silently never fires.
    """
    c = counts_1d.float()
    s = ExpertLoadTracker.shares(c)
    dead = (c if dead_counts is None else dead_counts.float()) == 0

    if alpha == 0.0:
        scales = torch.ones_like(s)
    else:
        safe_s = torch.where(dead, torch.ones_like(s), s)
        scales = safe_s.pow(alpha)
        # A dead expert has no signal this step; leave it at the uniform rate
        # unless freeze_dead is on, rather than letting the rule extrapolate.
        scales = torch.where(dead, torch.ones_like(scales), scales)

        alive = ~dead
        if normalize and bool(alive.any()):
            mean_alive = scales[alive].mean()
            if mean_alive > 0:
                scales = scales / mean_alive
        scales = scales.clamp(clamp_min, clamp_max)
        # Re-pin dead experts to the uniform rate: normalisation is defined over
        # the experts that actually received tokens, and a dead expert should sit
        # at 1.0 either way (or at 0.0 once freeze_dead is applied below).
        scales = torch.where(dead, torch.ones_like(scales), scales)

    if freeze_dead:
        scales = torch.where(dead, torch.zeros_like(scales), scales)
    return scales


class ExpertStatsWriter:
    """Append per-expert records to a JSONL file for offline correlation analysis.

    Per-expert series are far too wide for wandb (64 experts x 16 layers = 1024
    series per metric), so wandb gets the aggregates and this file gets the full
    joint distribution that the deliverable is computed from.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a")

    def write(self, record: dict):
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass
