"""Muon that accepts 3D expert tensors directly.

A `[E, out, in]` parameter is treated as a batch of E independent matrices: every
expert gets its own momentum slice, its own Frobenius normalisation and its own
Newton-Schulz orthogonalisation, computed with batched matmuls. The arithmetic per
matrix is exactly `torch.optim.Muon` (same coefficients, bf16 iteration, Nesterov
lerp, decoupled decay with the *unadjusted* lr, `sqrt(max(1, out/in))` lr adjust),
which `tests/test_muon.py` checks against the reference slice by slice.

This replaces the original trainer's 2D-proxy trick (one proxy Parameter per
expert, grads copied every step): no proxies, no copies, no Python loop over
experts, and the optimizer state is keyed by the real model parameters.

Per-expert learning rates ride on the same structure. A group may carry an
`lr_scale` vector of one multiplier per leading-dim slice, which scales that
expert's step *and* its decoupled weight decay -- so a multiplier of 0 freezes an
expert completely instead of letting weight decay keep shrinking it. The original
achieved this by writing `pg["lr"] *= scale` on each proxy's group; here it is a
group attribute, which is why `trainer.py` needs no per-expert code path.
`lr_scale=None` is the default and takes the unscaled code path byte for byte.

A group may also carry an `update_mask` (bool `[E]`): slices where it is False are
skipped *entirely* -- no momentum update, no decay, no step. That is the
difference from `lr_scale=0`, which still blends the (zero) gradient into the
momentum buffer; an expert accumulating gradient across several batches has to
keep its buffer frozen between events, or the buffer decays while it waits. The
masked path gathers the firing slices with `index_select`, runs Newton-Schulz on
those alone and scatters the result back, so the cost of a step is proportional to
the experts that actually fire rather than to E. `update_mask=None` is the default
and, like `lr_scale=None`, takes the original code path byte for byte.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

NS_COEFFS = (3.4445, -4.7750, 2.0315)


def zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7,
                               coeffs=NS_COEFFS) -> torch.Tensor:
    """Batched over all leading dims; acts on the last two."""
    a, b, c = coeffs
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    # Fused multiply-adds, as in torch.optim.Muon, so rounding matches the reference.
    mm_add = torch.addmm if X.dim() == 2 else torch.baddbmm
    for _ in range(steps):
        A = X @ X.mT
        B = mm_add(A, A, A, beta=b, alpha=c)
        X = mm_add(X, B, X, beta=a)
    return X.mT if transposed else X


def _broadcast_scale(scale: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Reshape a per-slice `lr_scale` [E] to broadcast over a `[E, out, in]` param.

    Only batched (3D) parameters can carry one: a 2D matrix is a single expert's
    worth of weights and there is nothing to index. Getting this wrong silently
    scales the wrong experts, so it raises instead.
    """
    if not isinstance(scale, torch.Tensor):
        scale = torch.as_tensor(scale)
    if p.dim() != 3:
        raise ValueError(f"lr_scale needs a 3D (batched) parameter, got {tuple(p.shape)}")
    if scale.dim() != 1 or scale.numel() != p.size(0):
        raise ValueError(f"lr_scale must be a 1D tensor of length {p.size(0)}, "
                         f"got {tuple(scale.shape)}")
    return scale.to(device=p.device, dtype=torch.float32).view(-1, 1, 1)


def _mask_indices(mask: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Validate a per-slice `update_mask` and return the indices it selects.

    Same contract as `_broadcast_scale`: batched parameters only, one entry per
    leading-dim slice. The dtype check is deliberate -- a float mask would select
    every slice through `nonzero`, silently stepping experts that were meant to be
    accumulating.
    """
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    if p.dim() != 3:
        raise ValueError(f"update_mask needs a 3D (batched) parameter, got {tuple(p.shape)}")
    if mask.dim() != 1 or mask.numel() != p.size(0):
        raise ValueError(f"update_mask must be a 1D tensor of length {p.size(0)}, "
                         f"got {tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        raise ValueError(f"update_mask must be a bool tensor, got {mask.dtype}")
    return mask.to(p.device).nonzero(as_tuple=False).flatten()


def adjusted_lr(lr: float, shape: torch.Size) -> float:
    out_dim, in_dim = shape[-2], shape[-1]
    return lr * math.sqrt(max(1.0, out_dim / in_dim))


class Muon(Optimizer):
    def __init__(self, params, lr=2e-2, weight_decay=0.1, momentum=0.95, nesterov=True,
                 ns_steps=5, eps=1e-7):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps, eps=eps,
                        lr_scale=None, update_mask=None)
        super().__init__(params, defaults)
        for g in self.param_groups:
            for p in g["params"]:
                if p.dim() not in (2, 3):
                    raise ValueError(f"Muon takes 2D or 3D (batched) params, got {tuple(p.shape)}")

    def _apply(self, dst, grad, buf, g, lr, wd, mom, s):
        """One Muon update, in place on `dst` (the weights) and `buf` (the momentum).

        Shared by the masked and unmasked paths so there is exactly one copy of the
        arithmetic: `dst` is the parameter itself when no mask is set, and a gathered
        copy of the firing slices when one is. `s` is an already-broadcast `[*,1,1]`
        lr multiplier, or None for the unscaled path.
        """
        buf.lerp_(grad, 1 - mom)
        update = grad.lerp(buf, mom) if g["nesterov"] else buf
        update = zeropower_via_newtonschulz(update, g["ns_steps"], g["eps"])
        if s is None:
            dst.mul_(1 - lr * wd)
            dst.add_(update, alpha=-adjusted_lr(lr, dst.shape))
        else:
            # float64 for the decay factor so that an all-ones lr_scale is
            # bit-identical to the scalar path above, which computes
            # `1 - lr * wd` in Python floats.
            dst.mul_((1.0 - s.double() * (lr * wd)).to(dst.dtype))
            # `update` is bf16 out of Newton-Schulz and `s` is fp32, so the
            # product promotes to fp32 -- the same arithmetic the scalar
            # `add_(update, alpha=...)` does, and the multiplier keeps its
            # precision instead of being rounded to bf16.
            dst.add_(update * s, alpha=-adjusted_lr(lr, dst.shape))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            lr, wd, mom = g["lr"], g["weight_decay"], g["momentum"]
            scale, mask = g.get("lr_scale"), g.get("update_mask")
            for p in g["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                # Validated against the full parameter, so the error message names
                # E rather than the number of slices that happen to be firing.
                s = None if scale is None else _broadcast_scale(scale, p)
                if mask is None:
                    # The momentum buffer is updated regardless of the multiplier:
                    # a frozen expert still sees (zero) gradients, exactly as a proxy
                    # parameter at lr=0 did in the original trainer. An `update_mask`
                    # is what stops the buffer instead.
                    self._apply(p, p.grad, buf, g, lr, wd, mom, s)
                    continue
                idx = _mask_indices(mask, p)
                if idx.numel() == 0:
                    continue
                sub_p = p.data.index_select(0, idx)
                sub_buf = buf.index_select(0, idx)
                self._apply(sub_p, p.grad.index_select(0, idx), sub_buf, g, lr, wd, mom,
                            None if s is None else s.index_select(0, idx))
                p.data.index_copy_(0, idx, sub_p)
                buf.index_copy_(0, idx, sub_buf)
        return loss
