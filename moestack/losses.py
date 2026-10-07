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

from .config import LossConfig
from .model import ModelOutput, RouterOut


@dataclass
class LossTerm:
    name: str
    value: torch.Tensor      # scalar, carries grad
    coef: float


def lm_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                           ignore_index=-100)


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
    terms = [LossTerm("lm", lm_cross_entropy(out.logits, targets), 1.0)]
    if rs:
        terms.append(LossTerm("aux", torch.stack([load_balance_loss(r) for r in rs]).float().mean(),
                              cfg.aux_coef))
        terms.append(LossTerm("z", torch.stack([router_z_loss(r) for r in rs]).mean(), cfg.z_coef))
    total = sum(t.coef * t.value for t in terms if t.coef != 0.0)
    return total, terms
