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


def adjusted_lr(lr: float, shape: torch.Size) -> float:
    out_dim, in_dim = shape[-2], shape[-1]
    return lr * math.sqrt(max(1.0, out_dim / in_dim))


class Muon(Optimizer):
    def __init__(self, params, lr=2e-2, weight_decay=0.1, momentum=0.95, nesterov=True,
                 ns_steps=5, eps=1e-7):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps, eps=eps)
        super().__init__(params, defaults)
        for g in self.param_groups:
            for p in g["params"]:
                if p.dim() not in (2, 3):
                    raise ValueError(f"Muon takes 2D or 3D (batched) params, got {tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            lr, wd, mom = g["lr"], g["weight_decay"], g["momentum"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - mom)
                update = grad.lerp(buf, mom) if g["nesterov"] else buf
                update = zeropower_via_newtonschulz(update, g["ns_steps"], g["eps"])
                p.mul_(1 - lr * wd)
                p.add_(update, alpha=-adjusted_lr(lr, p.shape))
        return loss
