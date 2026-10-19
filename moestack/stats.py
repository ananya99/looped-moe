"""Per-expert measurement: inequality of the routing, and how far experts move.

Pure functions of tensors -- no model, no optimizer, no distributed, no files --
so the statistics behind a claim can be tested on CPU. `callbacks.py` decides when
to call them and where the numbers go.

The quantities are the ones the expert-imbalance question is phrased in:

    gini / entropy      how unequal the token assignment is
    gini_alive          that inequality with deadness factored out, since
                        gini = dead_fraction + (1 - dead_fraction) * gini_alive
    spearman            rank correlation between an expert's token share and how
                        far its weights actually moved -- the deliverable
    delta_norms         per-expert ||w_after - w_before||_F, the "how far" above
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


def load_metrics(counts: torch.Tensor, per_layer: bool = True) -> dict:
    """Routing-balance metrics from exact global-batch counts `[L, E]`.

    `gini_alive` is the one that is not redundant: a rule that revives dead experts
    and one that evens out the working ones both lower `gini`, and only the split
    says which happened.
    """
    c = counts.float()
    if c.dim() == 1:
        c = c.unsqueeze(0)
    ginis, alives, deads, entropies, maxfracs, moms = [], [], [], [], [], []
    out = {}
    for li, row in enumerate(c):
        total = row.sum()
        if total <= 0:
            continue
        alive = row[row > 0]
        ginis.append(gini(row))
        alives.append(gini(alive) if alive.numel() > 1 else 0.0)
        deads.append(float((row == 0).float().mean()))
        entropies.append(normalized_entropy(row))
        maxfracs.append(float(row.max() / total))
        moms.append(float(row.max() / row.mean().clamp(min=1e-9)))
        if per_layer:
            # `load_per_layer/`, not `load/`: wandb sections panels by the metric
            # prefix, and 4 x n_layers series inside `load/` bury the eight
            # aggregates that are the ones you actually read every run.
            out[f"load_per_layer/gini/{li:02d}"] = ginis[-1]
            out[f"load_per_layer/gini_alive/{li:02d}"] = alives[-1]
            out[f"load_per_layer/dead_fraction/{li:02d}"] = deads[-1]
            out[f"load_per_layer/max_over_mean/{li:02d}"] = moms[-1]
    if not ginis:
        return {}
    n = len(ginis)
    out.update({
        "load/gini": sum(ginis) / n,
        "load/gini_alive": sum(alives) / n,
        "load/gini_max_layer": max(ginis),
        "load/dead_expert_fraction": sum(deads) / n,
        "load/dead_fraction_max_layer": max(deads),
        "load/normalized_entropy": sum(entropies) / n,
        "load/max_expert_fraction": sum(maxfracs) / n,
        "load/max_over_mean": sum(moms) / n,
    })
    return out


def delta_norms(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """Per-expert Frobenius norm of the step actually taken, `[E]` from `[E, ...]`."""
    return (after.float() - before.float()).flatten(1).norm(dim=1)


def row_norms(w: torch.Tensor) -> torch.Tensor:
    """Per-expert Frobenius norm of a weight tensor, `[E]` from `[E, ...]`."""
    return w.float().flatten(1).norm(dim=1)


def drift_metrics(counts: torch.Tensor, drift: torch.Tensor) -> dict:
    """Relate token share to distance travelled. `counts` and `drift` are `[L, E]`.

    `drift/dead_over_alive` is the zero-token-expert effect in one number: Muon
    renormalises a dead expert's decaying momentum back to unit norm, so it takes a
    full-magnitude step in a stale direction while weight decay shrinks it. A
    baseline sits near 1.0 (dead experts move as far as working ones); `freeze_dead`
    should drive it to 0.
    """
    c, d = counts.float(), drift.float()
    if c.dim() == 1:
        c, d = c.unsqueeze(0), d.unsqueeze(0)
    corr, corr_alive = [], []
    for row_c, row_d in zip(c, d):
        corr.append(spearman(row_c, row_d))
        alive = row_c > 0
        if int(alive.sum()) >= 3:
            corr_alive.append(spearman(row_c[alive], row_d[alive]))

    def _mean(xs):
        xs = [x for x in xs if not math.isnan(x)]
        return sum(xs) / len(xs) if xs else float("nan")

    dead, alive = c == 0, c > 0
    out = {
        "drift/corr_share_vs_update": _mean(corr),
        "drift/corr_share_vs_update_alive": _mean(corr_alive),
        "drift/mean": float(d.mean()),
        "drift/max": float(d.max()),
    }
    if bool(dead.any()) and bool(alive.any()):
        mean_alive = float(d[alive].mean())
        out["drift/dead_mean"] = float(d[dead].mean())
        out["drift/dead_over_alive"] = (float(d[dead].mean()) / mean_alive
                                        if mean_alive > 0 else float("nan"))
    return out
