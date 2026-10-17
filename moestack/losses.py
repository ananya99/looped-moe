"""Training objective as a sum of named terms.

    total = lm + aux_coef * mean_l(balance_l) + z_coef * mean_l(z_l)

Each term is returned separately so every one can be logged, and the coefficients
are independent by construction (no HF `router_aux_loss_coef` carrier). This
reproduces exactly what `imbalance_train.install_router_loss_patch` made HF compute,
with one documented difference: the balance term uses the router's own top-k
selection (fp32 probs) instead of re-running top-k on probs cast back to the
autocast dtype, which could pick a different expert on bf16 near-ties.

`lm` is the mean token cross-entropy over `targets` (all T positions). The original
trainer passed `labels=input_ids` to HF, which scored T-1 positions per sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import LossConfig
from .model import ModelOutput, RouterOut


@dataclass
class LossTerm:
    name: str
    value: torch.Tensor      # scalar, carries grad
    coef: float


def _ce_sum(flat_logits: torch.Tensor, flat_targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(flat_logits.float(), flat_targets, ignore_index=-100,
                           reduction="sum")


def lm_cross_entropy_sum(logits: torch.Tensor, targets: torch.Tensor,
                         chunk_rows: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """(summed cross-entropy, number of scored tokens).

    Training and evaluation both go through here, so they cannot drift apart on
    which positions count: -100 is ignored in the sum *and* excluded from the
    denominator.

    `chunk_rows > 0` scores the rows in chunks. When grad is required each chunk's
    fp32 cast and log-softmax are recomputed in backward rather than stored; under
    no_grad there is nothing to store and the chunking just caps the transient
    allocation. Either way the value is unchanged (up to fp32 summation order).
    At vocab 100k this, not the model, is what caps the batch size: 16x2048 rows
    cost 13 GB per fp32 copy, and the loss needs two of them.
    """
    flat = logits.reshape(-1, logits.size(-1))
    tgt = targets.reshape(-1)
    n_scored = (tgt != -100).sum()
    if chunk_rows <= 0 or flat.size(0) <= chunk_rows:
        return _ce_sum(flat, tgt), n_scored
    recompute = torch.is_grad_enabled() and flat.requires_grad
    total = flat.new_zeros((), dtype=torch.float32)
    for lo in range(0, flat.size(0), chunk_rows):
        rows, labels = flat[lo:lo + chunk_rows], tgt[lo:lo + chunk_rows]
        total = total + (checkpoint(_ce_sum, rows, labels, use_reentrant=False)
                         if recompute else _ce_sum(rows, labels))
    return total, n_scored


def lm_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                     chunk_rows: int = 0) -> torch.Tensor:
    """Mean token cross-entropy over `targets` (-100 ignored)."""
    total, n_scored = lm_cross_entropy_sum(logits, targets, chunk_rows)
    return total / n_scored.clamp(min=1)


def load_balance_loss(r: RouterOut) -> torch.Tensor:
    """Switch/Fedus: E * sum_e f_e * P_e, with f_e the fraction of the N*k
    assignments that went to expert e (sums to 1) and P_e the mean router prob."""
    E = r.logits.size(-1)
    probs = torch.softmax(r.logits.float(), dim=-1).to(r.logits.dtype)
    f = r.counts.float() / r.topk_idx.numel()
    return E * (f * probs.mean(dim=0)).sum()


def router_z_loss(r: RouterOut) -> torch.Tensor:
    return torch.logsumexp(r.logits.float(), dim=-1).pow(2).mean()


def compute_loss(out: ModelOutput, targets: torch.Tensor, cfg: LossConfig
                 ) -> tuple[torch.Tensor, list[LossTerm]]:
    rs = out.router_outs
    terms = [LossTerm("lm", lm_cross_entropy(out.logits, targets, cfg.lm_chunk_rows), 1.0)]
    if rs:
        terms.append(LossTerm("aux", torch.stack([load_balance_loss(r) for r in rs]).float().mean(),
                              cfg.aux_coef))
        terms.append(LossTerm("z", torch.stack([router_z_loss(r) for r in rs]).mean(), cfg.z_coef))
    total = sum(t.coef * t.value for t in terms if t.coef != 0.0)
    return total, terms
