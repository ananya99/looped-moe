"""Loss terms, Muon, parameter routing and the LR schedule."""

import math

import pytest
import torch

from moestack.config import LossConfig, OptimConfig
from moestack.losses import compute_loss, load_balance_loss, router_z_loss
from moestack.model import RouterOut, Transformer
from moestack.optim import Muon, build_optimizers, lr_at, tag_parameters

from conftest import tiny_args


# ── losses ───────────────────────────────────────────────────────────

def _router_out(logits, k):
    probs = logits.float().softmax(-1)
    w, idx = probs.topk(k, dim=-1)
    return RouterOut(logits, idx, w, torch.bincount(idx.flatten(), minlength=logits.size(-1)))


def test_balance_loss_is_one_when_perfectly_balanced():
    E, k = 8, 2
    # uniform probabilities; assignments spread exactly evenly over experts
    logits = torch.zeros(E * 4, E)
    idx = torch.stack([torch.arange(E * 4) % E, (torch.arange(E * 4) + 1) % E], dim=1)
    r = RouterOut(logits, idx, torch.full((E * 4, k), 0.5), torch.bincount(idx.flatten(), minlength=E))
    torch.testing.assert_close(load_balance_loss(r), torch.tensor(1.0))


def test_balance_loss_approaches_E_when_collapsed():
    E = 8
    logits = torch.full((64, E), -20.0)
    logits[:, 0] = 20.0
    r = _router_out(logits, 1)
    assert abs(load_balance_loss(r).item() - E) < 1e-3


def test_z_loss_value():
    logits = torch.randn(10, 8)
    torch.testing.assert_close(router_z_loss(_router_out(logits, 2)),
                               logits.logsumexp(-1).square().mean())


def test_zero_coefficients_drop_terms_but_still_report_them():
    torch.manual_seed(0)
    m = Transformer(tiny_args())
    x = torch.randint(0, 256, (2, 16))
    out = m(x)
    total, terms = compute_loss(out, x, LossConfig(aux_coef=0.0, z_coef=0.0))
    names = {t.name: t for t in terms}
    assert set(names) == {"lm", "aux", "z"}
    torch.testing.assert_close(total, names["lm"].value)


# ── Muon ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", [(4, 24, 16), (4, 16, 24), (3, 16, 16)])
def test_batched_muon_equals_torch_muon_per_slice(shape):
    torch.manual_seed(0)
    w3 = torch.nn.Parameter(torch.randn(*shape))
    slices = [torch.nn.Parameter(w3.detach()[i].clone()) for i in range(shape[0])]
    ours = Muon([w3], lr=0.02, weight_decay=0.1, momentum=0.95, nesterov=True)
    ref = torch.optim.Muon(slices, lr=0.02, weight_decay=0.1, momentum=0.95, nesterov=True)
    for step in range(4):
        g = torch.randn(*shape)
        if step == 2:
            g[1] = 0.0          # a zero-token expert: momentum still moves it, as in torch
        w3.grad = g.clone()
        for i, s in enumerate(slices):
            s.grad = g[i].clone()
        ours.step()
        ref.step()
        for i, s in enumerate(slices):
            # Newton-Schulz runs in bf16; batched vs per-matrix kernels may round one
            # bf16 ulp differently, i.e. up to ~lr * 2**-7 per element.
            torch.testing.assert_close(w3.detach()[i], s.detach(), atol=0.02 * 2**-7, rtol=0)


def test_muon_2d_equals_torch_muon():
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(32, 16))
    b = torch.nn.Parameter(a.detach().clone())
    oa, ob = Muon([a], lr=0.05), torch.optim.Muon([b], lr=0.05)
    for _ in range(3):
        g = torch.randn(32, 16)
        a.grad, b.grad = g.clone(), g.clone()
        oa.step(), ob.step()
    torch.testing.assert_close(a, b, atol=0.05 * 2**-7, rtol=0)


# ── parameter routing ───────────────────────────────────────────────

def test_parameter_roles_and_optimizer_table():
    m = Transformer(tiny_args())
    infos = tag_parameters(m)
    roles = {i.name: i.role for i in infos}
    assert roles["tok_embeddings.weight"] == "embed"
    assert roles["output.weight"] == "head"
    assert roles["layers.0.moe.router.weight"] == "router"
    assert roles["layers.0.moe.experts.gate_up_proj"] == "expert"
    assert roles["layers.1.attention.q_proj.weight"] == "hidden"
    assert roles["layers.1.attention.q_norm.weight"] == "norm"
    assert roles["norm.weight"] == "norm"

    opts = build_optimizers(m, OptimConfig(name="muon", lr=0.02))
    where = {}
    for name, o in opts.optimizers.items():
        for g in o.param_groups:
            for p in g["params"]:
                where[id(p)] = (name, g["lr_mult"], g["weight_decay"])
    named = dict(m.named_parameters())
    assert len(where) == len(named)           # every parameter exactly once
    assert where[id(named["layers.0.moe.experts.down_proj"])] == ("muon", 1.0, 0.1)
    assert where[id(named["layers.0.attention.o_proj.weight"])] == ("muon", 1.0, 0.1)
    assert where[id(named["layers.0.moe.router.weight"])] == ("adamw", 0.1, 0.01)
    assert where[id(named["tok_embeddings.weight"])] == ("adamw", 0.1, 0.01)
    assert where[id(named["layers.0.ffn_norm.weight"])] == ("adamw", 0.1, 0.01)

    opts.set_lr(0.01)
    assert opts.optimizers["muon"].param_groups[0]["lr"] == 0.01
    assert math.isclose(opts.optimizers["adamw"].param_groups[0]["lr"], 0.001)


def test_lr_schedule_matches_original():
    def original_get_lr(step, warmup_steps, n_steps, lr, min_lr):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))

    cfg = OptimConfig(lr=2e-2, min_lr_ratio=0.1, warmup_steps=100)
    for s in [0, 50, 99, 100, 1000, 1999]:
        base = lr_at(s, cfg, 2000)
        assert math.isclose(base, original_get_lr(s, 100, 2000, 2e-2, 2e-3))
        # the AdamW groups (lr_mult 0.1) follow the original parallel 1/10 curve
        assert math.isclose(0.1 * base, original_get_lr(s, 100, 2000, 2e-3, 2e-4))
