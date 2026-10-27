"""Compass alignment factors, measured but not applied.

ExpertMuon-Compass (Vaidya et al., arXiv:2610.04140) sets each expert's Muon step
length from how well its *orthogonalised* update agrees with its current gradient.
Muon normalises the update's singular values, so every expert matrix of a shape
moves about the same distance per step whatever its gradient looked like -- even an
expert whose update is poorly aligned with where the gradient now points, which is
what happens when the tokens routed to it change while its momentum still reflects
the old ones. That is the same premise as the expert-imbalance work here, reached
from the optimizer side rather than the routing side.

This module computes the factors **as a diagnostic**: the update stays plain Muon.
What you get is "how misaligned are my experts under the current optimizer", which
complements `grad_stats`: SNR asks whether an expert's gradient is trustworthy,
alignment asks whether the step Muon is about to take agrees with it.

The arithmetic, with `O_e` the orthogonalised direction and `G_e` the gradient
(paper eqs. 2-6; a family is the matching projection matrix across all experts of
one layer, so `gate_up_proj` and `down_proj` are separate families):

    cos(A, B) = <A, B>_F / (||A||_F ||B||_F)                              (2)
    a_e  = [cos(O_e, G_e)]_+                     expert alignment score
    u_e  = a_e / abar,  v_e = min(1, u_e),  c_e = v_e / vbar              (3)
    b_er = [cos(O_er, G_er)]_+                   per output row r
    q_er = min(1, b_er / bbar_e),  h_er = 1 + lam * (q_er - 1)            (4)
    a_e^radius = sqrt( sum_r h_er^2 ||O_er||^2 / ||O_e||_F^2 )            (5)

`lam = lr / peak_lr`, so the row factors relax to 1 as the schedule decays. The
family factor is capped at 1 before the second normalisation, which is what lets
`c_e` exceed 1 (small scores are pulled up by the mean) while the radius cannot:
it lies in [0, 1] by construction. Their product is the *expert scale*, which the
paper reports as 0.55-0.7 at the low end and 1.05-1.10 at the high end, with a mean
rising from 0.93 toward 1.0 as training proceeds.

Experts that received no tokens are excluded from the family means, as in the
paper -- their gradient is zero, so their "alignment" is not a measurement.
"""

from __future__ import annotations

import torch

EPS = 1e-12


def _cos(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    num = (a * b).sum(dim=dims)
    den = a.pow(2).sum(dim=dims).sqrt() * b.pow(2).sum(dim=dims).sqrt()
    return num / den.clamp(min=EPS)


def alignment(update: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """`a_e = [cos(O_e, G_e)]_+`, `[E]` from `[E, m, n]`."""
    return _cos(update.float(), grad.float(), dims=(-2, -1)).clamp(min=0.0)


def row_alignment(update: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """`b_er`, `[E, m]`: the same cosine taken row by row."""
    return _cos(update.float(), grad.float(), dims=-1).clamp(min=0.0)


def family_factor(a: torch.Tensor, alive: torch.Tensor | None = None) -> torch.Tensor:
    """`c_e` from the expert scores of one family (paper eq. 3).

    Two normalisations, and the order matters: dividing by the family mean puts an
    average expert at 1, the cap at 1 then stops an unusually well-aligned expert
    from dominating, and the second mean restores an average of 1 over the family
    -- so the factor redistributes step length rather than changing the total.
    """
    a = a.float()
    m = torch.ones_like(a, dtype=torch.bool) if alive is None else alive.bool()
    if not bool(m.any()):
        return torch.ones_like(a)
    u = a / a[m].mean().clamp(min=EPS)
    v = u.clamp(max=1.0)
    return v / v[m].mean().clamp(min=EPS)


def radius(update: torch.Tensor, b: torch.Tensor, lam: float) -> torch.Tensor:
    """`alpha_e` from the row scores (paper eqs. 4-5), in [0, 1].

    `alpha_e` is the norm of the row-weighted matrix over the norm of `O_e`, so it
    shortens the step without rotating it -- weighting the rows directly would
    change the direction Muon computed, which is the thing Compass preserves.
    """
    b = b.float()
    q = (b / b.mean(dim=-1, keepdim=True).clamp(min=EPS)).clamp(max=1.0)
    h = 1.0 + lam * (q - 1.0)
    row_sq = update.float().pow(2).sum(dim=-1)              # ||O_er||^2, [E, m]
    total = row_sq.sum(dim=-1).clamp(min=EPS)               # ||O_e||_F^2, [E]
    return ((h.pow(2) * row_sq).sum(dim=-1) / total).sqrt()


def expert_scale(update: torch.Tensor, grad: torch.Tensor, lam: float,
                 alive: torch.Tensor | None = None) -> dict:
    """The three `[E]` series for one family: `c_e`, `alpha_e` and their product."""
    a = alignment(update, grad)
    c = family_factor(a, alive)
    al = radius(update, row_alignment(update, grad), lam)
    return {"alignment": a, "family": c, "radius": al, "scale": c * al}
