"""Per-expert measurement: inequality of the routing, and how far experts move.

Pure functions of tensors -- no model, no optimizer, no distributed, no files --
so the statistics behind a claim can be tested on CPU. `callbacks.py` decides when
to call them and where the numbers go.

The quantities are the ones the expert-imbalance question is phrased in:

    gini / entropy      how unequal the token assignment is
    spearman            rank correlation between an expert's token share and how
                        far its weights actually moved -- the deliverable
    delta_norms         per-expert ||w_after - w_before||_F, the "how far" above
    summarize           the distribution of an [L, E] quantity, because a mean over
                        experts hides the one expert that is misbehaving

What the `*_metrics` functions return is deliberately narrow: a metric earns a
wandb series only if it names a decision, and everything else is a function of the
`[L, E]` matrices that `callbacks.ExpertStats` writes to the JSONL, to be
recomputed offline. `normalized_entropy` and `gini` on an arbitrary slice are kept
public for exactly those offline scripts, not because the dashboard calls them.
"""

from __future__ import annotations

import math

import torch


def gini(x: torch.Tensor) -> float:
    """Gini coefficient of a non-negative vector. 0 = uniform, ->1 = collapsed."""
    x = x.float().flatten().sort().values
    n = x.numel()
    if n == 0 or x.sum() <= 0:
        return 0.0
    idx = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    return float(((2 * idx - n - 1) * x).sum() / (n * x.sum()))


def normalized_entropy(counts: torch.Tensor) -> float:
    """Entropy of the assignment distribution over experts, in [0, 1]. 1 = uniform."""
    c = counts.float().flatten()
    total = c.sum()
    if total <= 0 or c.numel() < 2:
        return 0.0
    p = c / total
    nz = p[p > 0]
    return float(-(nz * nz.log()).sum() / math.log(c.numel()))


def _rank(v: torch.Tensor) -> torch.Tensor:
    """Ranks with ties averaged.

    Ties are the common case here -- many experts sit at exactly zero tokens, and
    Newton-Schulz makes many update norms near-identical. Breaking ties
    arbitrarily would invent an ordering and report correlation where there is none.
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


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rank correlation. NaN -- never 0.0 -- when either side has no rank
    variance, which happens often (all experts dead, or every update the same size)
    and must not be reported as "no correlation"."""
    x, y = x.float().flatten(), y.float().flatten()
    if x.numel() < 3 or x.numel() != y.numel():
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = rx.norm() * ry.norm()
    return float(rx @ ry / denom) if denom > 0 else float("nan")


def summarize(x: torch.Tensor, name: str, mask: torch.Tensor | None = None) -> dict:
    """Seven scalars for an `[L, E]` per-expert quantity.

    A mean over experts and then over layers is the right headline and the wrong
    alarm: one expert running away is invisible in it. These are what make a single
    cell visible -- `max` pulling away from `median` is the symptom, and
    `argmax_layer`/`argmax_expert` name the cell. The full matrix goes to the JSONL,
    which is what the thesis figure is drawn from; these are for watching a run.

    `argmax` is two integer series rather than one "05:17" string because wandb
    plots numbers and files strings as run metadata, where they draw nothing.

    `outlier_fraction` is judged per layer: depth changes the scale of every
    quantity here, so a cell is anomalous relative to its own layer's median rather
    than the model's. Median-absolute-deviation, not standard deviation, because
    the outlier being looked for would inflate the latter and so hide itself.

    `mask` selects which cells count (alive experts, say). An empty selection
    reports NaN -- never 0.0 -- for the same reason `spearman` does: "no data" and
    "measured zero" must not arrive as the same number.
    """
    v = x.float()
    if v.dim() == 1:
        v = v.unsqueeze(0)
    m = (torch.ones_like(v, dtype=torch.bool) if mask is None
         else mask.reshape(v.shape).bool())
    nan = float("nan")
    sel = v[m]
    if sel.numel() == 0:
        out = {f"{name}_{k}": nan for k in
               ("mean", "p10", "median", "p90", "max", "outlier_fraction")}
        # -1, not NaN: these are indices, and the NaN filter in WandbLogger would
        # drop a float NaN silently rather than show the cell is undefined.
        out[f"{name}_argmax_layer"] = -1
        out[f"{name}_argmax_expert"] = -1
        return out
    flat = v.masked_fill(~m, float("-inf")).flatten()
    arg = int(flat.argmax())
    out = {
        f"{name}_mean": float(sel.mean()),
        f"{name}_p10": float(sel.quantile(0.10)),
        f"{name}_median": float(sel.median()),
        f"{name}_p90": float(sel.quantile(0.90)),
        f"{name}_max": float(sel.max()),
        f"{name}_argmax_layer": arg // v.size(1),
        f"{name}_argmax_expert": arg % v.size(1),
    }
    n_out = n_tot = 0
    for row, row_mask in zip(v, m):
        r = row[row_mask]
        if r.numel() < 3:
            continue            # no usable median; counting it would dilute the rate
        med = r.median()
        dev = (r - med).abs()
        n_tot += r.numel()
        # MAD is zero whenever more than half a row shares one value, which is the
        # normal case and not a pathological one: Newton-Schulz drives many update
        # norms to near-identical values (see `_rank`). Taken literally that makes
        # the test `dev > 3 * 0` flag everything, and guarding on `mad > 0` instead
        # makes it flag nothing -- the alarm switching itself off exactly where a
        # lone outlier is most visible. Fall back to the mean deviation, which is
        # zero only for a genuinely constant row, and that row has no outlier.
        mad = dev.median()
        scale = mad if float(mad) > 0 else dev.mean()
        if float(scale) > 0:
            n_out += int((dev > 3 * scale).sum())
    out[f"{name}_outlier_fraction"] = (n_out / n_tot) if n_tot else nan
    return out


def load_metrics(counts: torch.Tensor) -> dict:
    """Routing-balance metrics from exact global-batch counts `[L, E]`.

    Two numbers, deliberately. Everything else that used to be here --
    `gini_alive`, `normalized_entropy`, `max_expert_fraction`, `max_over_mean`, the
    per-layer series and the worst-layer maxima -- is a pure function of the same
    `[L, E]` counts, which `ExpertStats` writes to the JSONL every cadence. They are
    recomputed offline when a run makes them worth looking at, instead of occupying
    a panel in every run that does not.

    (`max_over_mean` is DeepSeek's MaxVio + 1, so that is the one to recompute when
    positioning against published numbers.)
    """
    c = counts.float()
    if c.dim() == 1:
        c = c.unsqueeze(0)
    ginis, deads = [], []
    for row in c:
        if row.sum() <= 0:
            continue
        ginis.append(gini(row))
        deads.append(float((row == 0).float().mean()))
    if not ginis:
        return {}
    n = len(ginis)
    return {
        "load/gini": sum(ginis) / n,
        "load/dead_expert_fraction": sum(deads) / n,
    }


def delta_norms(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """Per-expert Frobenius norm of the step actually taken, `[E]` from `[E, ...]`."""
    return (after.float() - before.float()).flatten(1).norm(dim=1)


def row_norms(w: torch.Tensor) -> torch.Tensor:
    """Per-expert Frobenius norm of a weight tensor, `[E]` from `[E, ...]`."""
    return w.float().flatten(1).norm(dim=1)


def drift_metrics(counts: torch.Tensor, drift: torch.Tensor,
                  weights: torch.Tensor | None = None) -> dict:
    """Relate token share to distance travelled. All arguments are `[L, E]`.

    `drift/corr_share_vs_update` is the deliverable: does seeing more tokens
    actually buy more learning. `drift/dead_over_alive` is the zero-token-expert
    effect in one number -- Muon renormalises a dead expert's decaying momentum back
    to unit norm, so it takes a full-magnitude step in a stale direction while
    weight decay shrinks it. A baseline sits near 1.0 (dead experts move as far as
    working ones); `freeze_dead` should drive it to 0.

    `weights` are the per-expert weight norms. Given them, this also reports
    `update/rel` = ||dw_e|| / ||w_e||, summarized: the quantity Muon actually
    controls, and the only one comparable across layers and across training as the
    weights grow. Its `_max` / `_argmax_*` are the single-expert anomaly alarm.

    The raw `drift/mean`, `drift/max` and `drift/dead_mean` are gone: the full
    `drift` and `weight_norm_*` matrices are in the JSONL, `update/rel` is the
    normalised version worth watching, and an un-normalised norm is not comparable
    between two layers of different width anyway.
    """
    c, d = counts.float(), drift.float()
    if c.dim() == 1:
        c, d = c.unsqueeze(0), d.unsqueeze(0)
    corr = [spearman(row_c, row_d) for row_c, row_d in zip(c, d)]
    finite = [x for x in corr if not math.isnan(x)]
    out = {"drift/corr_share_vs_update": (sum(finite) / len(finite) if finite
                                          else float("nan"))}

    dead, alive = c == 0, c > 0
    if bool(dead.any()) and bool(alive.any()):
        mean_alive = float(d[alive].mean())
        out["drift/dead_over_alive"] = (float(d[dead].mean()) / mean_alive
                                       if mean_alive > 0 else float("nan"))
    if weights is not None:
        w = weights.float()
        if w.dim() == 1:
            w = w.unsqueeze(0)
        # Experts start at a non-zero norm and weight decay never reaches zero, so
        # a zero here means the tensor was not measured, not that the expert
        # vanished: mask rather than clamp, or the ratio reports a spurious max.
        out.update(summarize(d / w.clamp(min=1e-12), "update/rel", mask=w > 0))
    return out
