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


def routing_moments(router_outs, top_k: int) -> torch.Tensor:
    """`[3]` of (entropy, confidence, margin), meaned over tokens and layers.

    How *decisively* the router assigns, which is a different axis from how
    *evenly* it assigns: a router can be perfectly confident per token and
    perfectly balanced over experts (it sends 1/E of tokens to each, each one
    certainly), or maximally unsure per token and still badly imbalanced. The
    `load/*` metrics cannot see this, including the normalised count entropy that
    used to live there -- that is the entropy of the assignment *histogram*, not of
    the router's per-token distribution.

    - `entropy`    mean over tokens of H(softmax(logits)), in **nats, unnormalised**,
                   to be directly comparable with published numbers (Compass
                   2610.04140 Fig. 3c reports 0.786-1.02 nats at E=8; the ceiling is
                   log E, so 4.159 at E=64). Divide by log E for a [0, 1] version.
    - `confidence` mean top-1 probability.
    - `margin`     mean gap between the k-th and (k+1)-th probability: how close the
                   router came to picking a different expert. Small margins mean the
                   assignment is unstable even when the entropy looks settled.

    Computed from the full softmax, not the renormalised top-k weights: that is what
    the router z-loss acts on (it penalises logsumexp of these logits), so it is the
    distribution whose shape the z-loss coefficient is actually controlling.
    """
    ents, confs, margins = [], [], []
    for r in router_outs:
        p = r.logits.detach().float().softmax(-1)
        ents.append(-(p * p.clamp(min=1e-12).log()).sum(-1).mean())
        top = p.topk(min(top_k + 1, p.size(-1)), dim=-1).values
        confs.append(top[:, 0].mean())
        margins.append((top[:, top_k - 1] - top[:, top_k]).mean()
                       if top.size(-1) > top_k else top.new_zeros(()))
    return torch.stack([torch.stack(ents).mean(), torch.stack(confs).mean(),
                        torch.stack(margins).mean()])


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
    ginis, deads, vios = [], [], []
    for row in c:
        if row.sum() <= 0:
            continue
        ginis.append(gini(row))
        deads.append(float((row == 0).float().mean()))
        mean = row.mean().clamp(min=1e-9)
        vios.append(float((row.max() - mean) / mean))
    if not ginis:
        return {}
    n = len(ginis)
    return {
        "load/gini": sum(ginis) / n,
        "load/dead_expert_fraction": sum(deads) / n,
        # MaxVio (Wang et al. 2024, as reported by ExpertMuon-Compass 2610.04140
        # Fig. 3b): the relative excess load of the most-used expert, averaged over
        # the MoE layers. **Zero means an even load** -- which is why it is
        # `max/mean - 1` and not `max/mean`. Kept despite being a function of the
        # counts, because it is the one load number directly comparable with
        # published figures; Compass reports it falling ~0.2 -> ~0.05 over training.
        "load/maxvio": sum(vios) / n,
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
    `drift/rel` = ||dw_e|| / ||w_e||, summarized: the quantity Muon actually
    controls, and the only one comparable across layers and across training as the
    weights grow. Its `_max` / `_argmax_*` are the single-expert anomaly alarm.

    **`drift/rel`, not `update/rel`.** Both are ||dw||/||w||, and they are different
    measurements: this one is net displacement over the whole reporting interval on
    the step clock, while `callbacks.ExpertGradStats` publishes `update/rel` for the
    displacement of one *event* on the event clock. Under one name the later writer
    won silently and the series alternated between the two. The namespace now says
    which clock: `drift/*` is the interval, `update/*` is the event. Runs logged
    before this split carry the interval quantity under `update/rel_*`.

    The raw `drift/mean`, `drift/max` and `drift/dead_mean` are gone: the full
    `drift` and `weight_norm_*` matrices are in the JSONL, `drift/rel` is the
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
        out.update(summarize(d / w.clamp(min=1e-12), "drift/rel", mask=w > 0))
    return out
