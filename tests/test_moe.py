"""MoE layer invariants: dispatch/combine, expert implementations, routing counts."""

import torch
import torch.nn.functional as F

from moestack.model import MoE, LocalDispatcher, GroupedExperts
from moestack.initialization import init_weights
from moestack.model import Transformer

from conftest import tiny_args


def _moe(**kw):
    torch.manual_seed(0)
    args = tiny_args(**kw)
    m = MoE(args)
    torch.nn.init.normal_(m.router.weight, std=0.5)
    torch.nn.init.normal_(m.experts.gate_up_proj, std=0.1)
    torch.nn.init.normal_(m.experts.down_proj, std=0.1)
    return m, args


def naive_moe(m: MoE, x):
    """Per token, per selected expert. The definition of the layer."""
    B, T, D = x.shape
    flat = x.reshape(-1, D)
    r = m.router(flat)
    out = torch.zeros_like(flat)
    for n in range(flat.size(0)):
        for j in range(r.topk_idx.size(1)):
            e = int(r.topk_idx[n, j])
            gate, up = F.linear(flat[n], m.experts.gate_up_proj[e]).chunk(2)
            out[n] += r.topk_weight[n, j] * F.linear(F.silu(gate) * up, m.experts.down_proj[e])
    return out.view(B, T, D)


def test_moe_matches_naive_definition():
    for impl in ("grouped_mm", "loop"):
        m, args = _moe(experts_impl=impl)
        x = torch.randn(2, 5, args.dim)
        y, _ = m(x)
        torch.testing.assert_close(y, naive_moe(m, x), atol=1e-5, rtol=1e-4)


def test_grouped_mm_equals_loop_forward_and_backward():
    m, args = _moe(experts_impl="grouped_mm")
    x = torch.randn(3, 7, args.dim)
    y1, _ = m(x)
    y1.square().sum().backward()
    g1 = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    m.experts.impl = "loop"
    y2, _ = m(x)
    y2.square().sum().backward()
    torch.testing.assert_close(y1, y2, atol=1e-5, rtol=1e-4)
    for a, b in zip(g1, (p.grad for p in m.parameters())):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_dispatch_conserves_tokens_and_counts():
    m, args = _moe()
    flat = torch.randn(40, args.dim)
    r = m.router(flat)
    rows, tpe, h = LocalDispatcher(args.n_experts).dispatch(flat, r)
    assert rows.size(0) == 40 * args.top_k == int(tpe.sum())
    assert torch.equal(tpe, r.counts)
    assert torch.equal(r.counts, torch.bincount(r.topk_idx.flatten(), minlength=args.n_experts))
    # rows are grouped by expert in order, and each row's token really chose that expert
    assert torch.equal(rows, flat[h.token_idx])
    expert_of_row = torch.repeat_interleave(torch.arange(args.n_experts), tpe)
    chose = (r.topk_idx[h.token_idx] == expert_of_row[:, None]).any(dim=1)
    assert chose.all()


def test_combine_with_identity_experts_returns_gate_weighted_input():
    m, args = _moe()
    flat = torch.randn(30, args.dim)
    r = m.router(flat)
    d = LocalDispatcher(args.n_experts)
    rows, _, h = d.dispatch(flat, r)
    out = d.combine(rows, h)
    torch.testing.assert_close(out, flat * r.topk_weight.sum(-1, keepdim=True))


def test_token_permutation_equivariance():
    m, args = _moe()
    x = torch.randn(1, 12, args.dim)
    perm = torch.randperm(12)
    y, _ = m(x)
    yp, _ = m(x[:, perm])
    torch.testing.assert_close(yp, y[:, perm], atol=1e-6, rtol=1e-5)


def test_norm_topk_prob_renormalises():
    m, args = _moe(arch="qwen3moe")
    assert args.norm_topk_prob
    r = m.router(torch.randn(20, args.dim))
    torch.testing.assert_close(r.topk_weight.sum(-1), torch.ones(20))


def test_every_parameter_gets_a_gradient():
    torch.manual_seed(0)
    model = Transformer(tiny_args())
    init_weights(model)
    x = torch.randint(0, model.args.vocab_size, (4, 32))
    out = model(x)
    # Use all outputs + router logits so even a never-selected expert's router row is reached.
    (out.logits.float().logsumexp(-1).mean() + sum(r.logits.float().square().mean() for r in out.router_outs)).backward()
    for n, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, n
    # Each expert that received tokens has a non-zero gradient slice.
    for layer, r in zip(model.layers, out.router_outs):
        g = layer.moe.experts.gate_up_proj.grad.flatten(1).norm(dim=1)
        assert torch.equal(g > 0, r.counts > 0)
