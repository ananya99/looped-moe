"""Per-expert learning-rate rules: token counts -> one LR multiplier per expert.

The experiment behind this: Muon's Newton-Schulz step normalises the update's
singular values, so scaling an expert's *gradient* does nothing -- only its
*learning rate* survives the normalisation. Per-expert LR is therefore the only
handle on "let experts that saw few tokens learn differently", and it is applied
as a multiplier vector inside the optimizer (see `optim/muon.py`), not as a
gradient hook.

Everything here is a pure function of a `[L, E]` count matrix, which is what
`StepState.expert_counts` already is: exact top-k counts, summed over every
micro-batch of the accumulation window and every DDP rank. Nothing in this file
touches the model, the optimizer or `torch.distributed`.

`tests/test_per_expert_lr.py` checks these rules against `compute_lr_scales` in
the original `expert_stats.py`, layer by layer.
"""

from __future__ import annotations

import torch


def token_shares(counts: torch.Tensor) -> torch.Tensor:
    """Token share `s_e = n_e / mean(n)` per row: 1.0 is a perfectly balanced expert.

    A row whose experts saw no tokens at all has no share to speak of; it gets
    ones, so the downstream rule degenerates to uniform rather than to NaN.
    """
    c = counts.float()
    mean = c.mean(dim=-1, keepdim=True)
    return torch.where(mean > 0, c / mean.clamp_min(1e-9), torch.ones_like(c))


def ema_update(prev: torch.Tensor | None, counts: torch.Tensor, decay: float) -> torch.Tensor:
    """`decay * prev + (1 - decay) * counts`, seeded by the first observation.

    Instantaneous counts are very noisy exactly where `n_e` is small, which is the
    regime this experiment is about; the EMA is what a smoothed rule runs on.
    Deadness, though, stays keyed on the raw counts -- see `lr_scales_from_counts`.
    """
    c = counts.float()
    return c if prev is None else decay * prev.to(c) + (1.0 - decay) * c


def lr_scales_from_counts(
    counts: torch.Tensor,
    alpha: float = 0.0,
    freeze_dead: bool = False,
    normalize: bool = True,
    clamp_min: float = 1e-3,
    clamp_max: float = 10.0,
    dead_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-expert LR multipliers `clamp(s_e ** alpha)`, one row per layer.

        alpha =  0     uniform -- multipliers are exactly 1.0 and this is the baseline
        alpha =  1     token-proportional: restores the SGD-like scaling that
                       Newton-Schulz normalisation removes
        alpha =  0.5   the square-root compromise
        alpha = -0.5   the other sign: compensate cold experts *upward*

    Four details carried over from `expert_stats.compute_lr_scales`, each of which
    changes the experiment if dropped:

    - Dead experts are resolved *before* exponentiation, because `0 ** -0.5` is inf.
      A dead expert has no signal this step, so the rule does not extrapolate: it
      sits at 1.0, or at 0.0 once `freeze_dead` is on.
    - `normalize` rescales to mean 1.0 **over the experts that got tokens**, so
      alpha redistributes the learning rate without also moving the layer's average
      rate. Without it an alpha sweep confounds redistribution with an overall rate
      change.
    - The clamp is deliberately asymmetric: a floor far below 1 only means an expert
      learns slowly, whereas a high ceiling multiplies Muon's LR on a hot expert and
      can destabilise it.
    - `dead_counts` decides *who is dead* separately from what drives the share.
      When `counts` is an EMA, an expert that received nothing this step still
      carries EMA mass from earlier steps, so freezing has to be keyed on the raw
      per-step counts or it silently never fires.

    `freeze_dead` sets the multiplier to exactly 0. Both Muon and AdamW scale
    decoupled weight decay by the learning rate, so a 0 multiplier suppresses the
    weight-decay shrinkage *and* the stale-momentum step together -- which is the
    whole point of that control.
    """
    c = counts.float()
    raw = c if dead_counts is None else dead_counts.float().to(c)
    if raw.shape != c.shape:
        raise ValueError(f"dead_counts {tuple(raw.shape)} must match counts {tuple(c.shape)}")
    dead = raw == 0
    ones = torch.ones_like(c)

    if alpha == 0.0:
        scales = ones.clone()
    else:
        s = token_shares(c)
        scales = torch.where(dead, ones, s).pow(alpha)
        scales = torch.where(dead, ones, scales)
        if normalize:
            alive = (~dead).to(c)
            n_alive = alive.sum(dim=-1, keepdim=True)
            mean_alive = (scales * alive).sum(dim=-1, keepdim=True) / n_alive.clamp_min(1)
            scales = torch.where(mean_alive > 0, scales / mean_alive.clamp_min(1e-9), scales)
        scales = scales.clamp(clamp_min, clamp_max)
        # Re-pin after the clamp: normalisation is defined over the alive experts,
        # and a dead one belongs at 1.0 either way (or at 0.0, just below).
        scales = torch.where(dead, ones, scales)

    if freeze_dead:
        scales = torch.where(dead, torch.zeros_like(scales), scales)
    return scales


def scale_summary(scales: torch.Tensor, counts: torch.Tensor,
                  base_lr: float | None = None) -> dict:
    """Aggregates for wandb. The full `[L, E]` matrix is far too wide to log, so
    the dashboard gets the spread and the joint distribution goes to a file.

    `lr_scale/*` are the multipliers themselves, i.e. each expert's rate *relative
    to the base LR* (1.0 = the rate it would have had without the rule). Passing
    `base_lr` adds `expert_lr/*` in absolute units, directly comparable with the
    `train/lr` curve.

    Those absolute numbers are still one step short of what Muon applies: its
    `adjusted_lr` multiplies by `sqrt(max(1, out/in))` **per parameter shape**, so
    an expert's `down_proj` and `gate_up_proj` already learn at different rates
    before any per-expert multiplier. Reporting that split per expert would double
    every series for a factor the rule does not touch, so it stays out.
    """
    s, c = scales.float(), counts.float()
    alive = c > 0
    n_alive = int(alive.sum())
    alive_scales = s[alive] if n_alive else s.new_zeros(0)
    nan = float("nan")
    out = {
        "lr_scale/min": float(s.min()) if s.numel() else nan,
        "lr_scale/max": float(s.max()) if s.numel() else nan,
        "lr_scale/mean_alive": float(alive_scales.mean()) if n_alive else nan,
        "lr_scale/max_alive": float(alive_scales.max()) if n_alive else nan,
        "lr_scale/min_alive": float(alive_scales.min()) if n_alive else nan,
        # How far apart the rule drives the fastest and slowest working expert.
        # 1.0 means it is doing nothing; this is the one number that says whether
        # an alpha sweep is actually separating the cells.
        "lr_scale/spread_alive": (float(alive_scales.max() / alive_scales.min())
                                  if n_alive and float(alive_scales.min()) > 0 else nan),
        "lr_scale/frozen_fraction": float((s == 0).float().mean()) if s.numel() else nan,
    }
    if base_lr is not None:
        out.update({
            "expert_lr/min": out["lr_scale/min"] * base_lr,
            "expert_lr/max": out["lr_scale/max"] * base_lr,
            "expert_lr/mean_alive": out["lr_scale/mean_alive"] * base_lr,
            "expert_lr/base": base_lr,      # what every expert would get at alpha=0
        })
    return out
