"""Per-expert gradient quality: is an expert's update signal, or noise that
happened not to cancel?

Within one window, expert e's gradient is a sum over micro-batches,
`g_e = sum_i g_e,i`. For a cold expert most `g_e,i` are exactly zero -- it saw no
tokens in that micro-batch -- and the few that are not point in near-random
directions, each driven by whichever handful of tokens happened to route there. So
the sum is dominated by accident. The signal-to-noise ratio says exactly that:

    var_e = (1/M) sum_j ||g_e,j||^2 - ||gbar_e||^2        (spread per micro-batch)
    SNR_e = ||gbar_e|| / sqrt(var_e / M)                  (mean vs its own error)

**The `/ M` is the whole point, and is easy to get wrong.** The obvious reading --
mean over micro-batches divided by their spread, `||gbar|| / sqrt(var)` -- is the
*per-sample* ratio, and it is flat in M by construction: as the window widens
`gbar` converges to the true mean and `var` to the noise variance, so the ratio
converges to a constant that says nothing about window size. What the optimizer
actually steps on is `gbar`, whose error is the standard error `sqrt(var / M)`, not
the standard deviation. That version grows as `sqrt(M)`, which is the prediction
expert-local accumulation is built on: average more micro-batches, get a direction
you can trust further.

It also behaves better at the cold end. An expert that fires in exactly one
micro-batch out of M reads `sqrt(M / (M - 1))`, i.e. about 1.0 whatever M is -- one
sample, so the mean is worth about as much as its own uncertainty. Under the
per-sample ratio the same expert would read `1 / sqrt(M - 1)` and appear to get
*worse* as the window lengthened, when nothing about it had changed.

**Why this is cheap.** Recovering `g_e,i` looks like it needs the per-micro-batch
gradient, which the training loop never materialises: `(loss / accum).backward()`
adds into `p.grad` and the terms are gone. Differencing successive `p.grad`
snapshots would cost a buffer the size of every expert gradient. It is not
necessary. Experts are single 3D `[E, ...]` tensors, so a **leaf grad hook** sees
that backward's contribution before it is accumulated -- 32 hooks at depth 16, each
reducing immediately to an `[E]` vector of squared norms. The identity above then
needs only two running `[L, E]` scalars, because

    sum_j ||g_j||^2   comes from the hooks, and
    ||sum_j g_j||     is read off `p.grad` when the window closes.

Both are separable across an expert's two matrices, so `gate_up` and `down` can be
accumulated into the same array without pairing them per micro-batch:
`||g||^2 = ||g_gate||^2 + ||g_down||^2` either way.

Scale cancels in the ratio, which is why `(loss / accum)` scaling and Muon's own
normalisation can be ignored here. What does *not* cancel is mixing per-rank and
reduced quantities -- see `close()`.
"""

from __future__ import annotations

import torch

from .parallel import all_reduce_


def _expert_sq_norms(g: torch.Tensor) -> torch.Tensor:
    """`[E]` squared Frobenius norms from an `[E, ...]` gradient."""
    return g.detach().float().flatten(1).pow(2).sum(1)


class ExpertGradMoments:
    """Running first and second moments of the per-micro-batch expert gradients.

    Lifecycle per window: `attach` once, then `tick()` on every micro-batch, then
    `close()` when the window ends, which returns the statistics and resets the
    experts that fired. Windows are per expert, so `micro` is `[L, E]` rather than
    a scalar -- under expert-local accumulation two experts in the same layer can
    be several micro-batches apart.
    """

    def __init__(self, n_layers: int, n_experts: int, device=None):
        self.sumsq = torch.zeros(n_layers, n_experts, dtype=torch.float32, device=device)
        self.micro = torch.zeros(n_layers, n_experts, dtype=torch.long, device=device)
        self._handles: list = []
        self._armed = False

    # -- wiring --------------------------------------------------------
    def attach(self, moe_layers) -> "ExpertGradMoments":
        """Hook both expert matrices of every layer.

        The hook is on the *parameter*, not the module, so it is indifferent to the
        DDP and compile wrappers: `parallelize` returns a wrapper to call forward
        on, but `model` keeps owning these tensors.
        """
        self.detach()
        for li, moe in enumerate(moe_layers):
            for p in (moe.experts.gate_up_proj, moe.experts.down_proj):
                self._handles.append(p.register_hook(self._make_hook(li)))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _make_hook(self, layer: int):
        def hook(grad):
            # Disarmed during anything that is not a training micro-batch, so an
            # eval or a probe backward cannot enter the statistics.
            if self._armed:
                self.sumsq[layer] += _expert_sq_norms(grad).to(self.sumsq.device)
            return grad          # never modify the gradient
        return hook

    def arm(self):
        self._armed = True

    def disarm(self):
        self._armed = False

    # -- the clock -----------------------------------------------------
    def tick(self, mask: torch.Tensor | None = None):
        """One micro-batch has entered every open window."""
        if mask is None:
            self.micro += 1
        else:
            self.micro += mask.to(self.micro.device).long()

    def close(self, grad_sq: torch.Tensor, world_size: int = 1) -> dict:
        """Statistics for the windows that are ending.

        `grad_sq` is `[L, E]`: the squared norm of the *accumulated* gradient each
        expert is about to step on, both matrices together (see
        `accumulated_grad_sq`).

        The DDP correction is the one place scale matters. The hooks fire on local
        gradients, before the reduction, so `sumsq` is this rank's contribution and
        summing it over ranks gives `sum_j ||g_j||^2` over every (rank,
        micro-batch) sample. But `p.grad` has been *averaged* over ranks, so the
        matching numerator is `world_size * p.grad`, i.e. `world_size**2 * grad_sq`.
        Mixing the two -- a local second moment against a reduced first moment --
        would silently scale SNR by the world size, and a 1-GPU debug run would not
        show it.
        """
        M = (self.micro * world_size).float()
        sumsq = all_reduce_(self.sumsq.clone())
        nan = torch.full_like(sumsq, float("nan"))

        # ||gbar||^2, with gbar the mean over all samples in the window.
        mean_sq = (grad_sq.to(sumsq) * world_size ** 2) / M.clamp(min=1) ** 2
        var = sumsq / M.clamp(min=1) - mean_sq
        # Standard error of the mean, not the standard deviation -- see the module
        # docstring. This factor is what makes the metric answer "how much can this
        # update be trusted" rather than "how noisy is one micro-batch".
        stderr = (var.clamp(min=0) / M.clamp(min=1)).sqrt()
        # M < 2 has no spread to measure, and var <= 0 is float cancellation at the
        # same place. NaN, not 0.0 or inf: "not measurable" must not arrive looking
        # like a measurement, the same contract as `stats.spearman`.
        ok = (M >= 2) & (var > 0)
        snr = torch.where(ok, mean_sq.clamp(min=0).sqrt() / stderr.clamp(min=1e-30), nan)
        return {
            "snr": snr,
            "grad_norm": torch.where(M > 0, mean_sq.clamp(min=0).sqrt(), nan),
            "micro": self.micro.clone(),
        }

    def reset(self, mask: torch.Tensor | None = None):
        """Clear the windows that just closed; `None` clears all of them."""
        if mask is None:
            self.sumsq.zero_()
            self.micro.zero_()
        else:
            m = mask.to(self.sumsq.device).bool()
            self.sumsq[m] = 0.0
            self.micro[m] = 0


def accumulated_grad_sq(moe_layers) -> torch.Tensor:
    """`[L, E]` squared norm of each expert's accumulated gradient, both matrices
    together -- matched to how `sumsq` combines them, so the ratio in `close()` is
    like for like. Experts whose gradient is None (nothing has flowed yet) read 0.
    """
    rows = []
    for moe in moe_layers:
        total = None
        for p in (moe.experts.gate_up_proj, moe.experts.down_proj):
            if p.grad is None:
                continue
            sq = _expert_sq_norms(p.grad)
            total = sq if total is None else total + sq
        rows.append(total if total is not None
                    else torch.zeros(moe.experts.gate_up_proj.size(0),
                                     device=moe.experts.gate_up_proj.device))
    return torch.stack(rows)
