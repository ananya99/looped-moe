"""Expert-local gradient accumulation.

Stage 1: the mask inside Muon (moestack/optim/muon.py) and the addressing that
writes it (moestack/optim/builder.py). An expert that is still filling its
accumulation window must be left *completely* alone -- weights, decay and momentum
-- which is exactly what `lr_scale=0` does not give you.
"""

import pytest
import torch

from moestack.optim import Muon, build_optimizers

MUON_KW = dict(lr=0.02, weight_decay=0.1, momentum=0.95, nesterov=True)
E, OUT, IN = 4, 24, 16
ALL = torch.ones(E, dtype=torch.bool)


def _grads(n_steps, seed=0, e=E):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(e, OUT, IN, generator=g) for _ in range(n_steps)]


def _run(param, grads, mask=None, scales=None, **kw):
    opt = Muon([param], **{**MUON_KW, **kw})
    if mask is not None:
        opt.param_groups[0]["update_mask"] = mask
    if scales is not None:
        opt.param_groups[0]["lr_scale"] = scales
    for g in grads:
        param.grad = g.clone()
        opt.step()
    return param, opt


# ── the baseline guarantee ───────────────────────────────────────────

def test_an_all_true_mask_is_bit_identical_to_no_mask():
    """The gather/scatter path has to be exact, not close: every accumulation run
    spends most of its steps with some experts firing, and a run with the feature
    off has to reproduce one built before it existed."""
    grads, w0 = _grads(5), torch.randn(E, OUT, IN)
    plain, _ = _run(torch.nn.Parameter(w0.clone()), grads)
    masked, _ = _run(torch.nn.Parameter(w0.clone()), grads, mask=ALL.clone())
    assert torch.equal(plain.data, masked.data)


def test_an_empty_mask_touches_nothing_at_all():
    w0 = torch.randn(E, OUT, IN)
    p, opt = _run(torch.nn.Parameter(w0.clone()), _grads(3),
                  mask=torch.zeros(E, dtype=torch.bool))
    assert torch.equal(p.data, w0)
    assert "momentum_buffer" in opt.state[p]          # allocated, never written
    assert opt.state[p]["momentum_buffer"].abs().sum() == 0


# ── what the mask actually does ──────────────────────────────────────

def test_masked_slices_keep_their_weights_and_their_momentum_exactly():
    grads, w0 = _grads(4, seed=1), torch.randn(E, OUT, IN)
    mask = torch.tensor([False, True, False, True])
    p, opt = _run(torch.nn.Parameter(w0.clone()), grads, mask=mask)
    buf = opt.state[p]["momentum_buffer"]
    for e in range(E):
        if mask[e]:
            assert not torch.allclose(p.data[e], w0[e]), f"expert {e} should have moved"
            assert buf[e].abs().sum() > 0
        else:
            assert torch.equal(p.data[e], w0[e]), f"expert {e} moved"
            assert buf[e].abs().sum() == 0, f"expert {e}'s momentum decayed while waiting"


def test_a_masked_expert_steps_as_if_it_had_just_started():
    """The whole difference from `lr_scale=0`. There, the buffer keeps lerping
    towards the zero gradients an idle expert sees, so by the time it is unfrozen
    its direction is a blend with that idling. Here the buffer is frozen, so the
    step it eventually takes is built from its own gradient alone."""
    grads, w0 = _grads(3, seed=7), torch.randn(E, OUT, IN)
    p = torch.nn.Parameter(w0.clone())
    opt = Muon([p], **MUON_KW)
    opt.param_groups[0]["update_mask"] = torch.tensor([False, True, True, True])
    for g in grads[:-1]:
        p.grad = g.clone()
        opt.step()
    assert torch.equal(p.data[0], w0[0])

    opt.param_groups[0]["update_mask"] = ALL.clone()
    p.grad = grads[-1].clone()
    opt.step()

    fresh, _ = _run(torch.nn.Parameter(w0[:1].clone()), [grads[-1][:1]])
    torch.testing.assert_close(p.data[0], fresh.data[0])


def test_a_firing_slice_matches_the_same_expert_stepped_on_its_own():
    """Gathering must not change the arithmetic of the experts that do fire."""
    grads, w0 = _grads(3, seed=2), torch.randn(E, OUT, IN)
    mask = torch.tensor([False, True, False, True])
    p, _ = _run(torch.nn.Parameter(w0.clone()), grads, mask=mask)
    for e in mask.nonzero().flatten().tolist():
        ref, _ = _run(torch.nn.Parameter(w0[e:e + 1].clone()),
                      [g[e:e + 1] for g in grads])
        torch.testing.assert_close(p.data[e], ref.data[0], msg=f"expert {e}")


def test_the_mask_composes_with_a_per_expert_lr_scale():
    grads, w0 = _grads(2, seed=3), torch.randn(E, OUT, IN)
    mask = torch.tensor([True, False, True, False])
    scales = torch.tensor([0.5, 2.0, 1.0, 3.0])
    p, _ = _run(torch.nn.Parameter(w0.clone()), grads, mask=mask, scales=scales)
    for e in (0, 2):
        ref, _ = _run(torch.nn.Parameter(w0[e:e + 1].clone()),
                      [g[e:e + 1] for g in grads],
                      lr=MUON_KW["lr"] * float(scales[e]))
        torch.testing.assert_close(p.data[e], ref.data[0], msg=f"expert {e}")
    assert torch.equal(p.data[1], w0[1]) and torch.equal(p.data[3], w0[3])


def test_mask_survives_an_optimizer_state_dict_round_trip():
    mask = torch.tensor([True, False, True, False])
    p = torch.nn.Parameter(torch.randn(E, OUT, IN))
    opt = Muon([p], **MUON_KW)
    opt.param_groups[0]["update_mask"] = mask
    opt2 = Muon([torch.nn.Parameter(torch.randn(E, OUT, IN))], **MUON_KW)
    opt2.load_state_dict(opt.state_dict())
    assert torch.equal(opt2.param_groups[0]["update_mask"], mask)


# ── misuse fails loudly ──────────────────────────────────────────────

@pytest.mark.parametrize("mask, match", [
    (torch.ones(E + 1, dtype=torch.bool), "1D tensor of length"),
    (torch.ones(E), "bool tensor"),                     # a float mask selects everything
    (torch.ones(E, dtype=torch.long), "bool tensor"),
])
def test_a_malformed_mask_is_an_error(mask, match):
    p = torch.nn.Parameter(torch.randn(E, OUT, IN))
    opt = Muon([p], **MUON_KW)
    opt.param_groups[0]["update_mask"] = mask
    p.grad = torch.randn(E, OUT, IN)
    with pytest.raises(ValueError, match=match):
        opt.step()


def test_a_mask_on_a_2d_parameter_is_an_error():
    p = torch.nn.Parameter(torch.randn(OUT, IN))
    opt = Muon([p], **MUON_KW)
    opt.param_groups[0]["update_mask"] = torch.ones(OUT, dtype=torch.bool)
    p.grad = torch.randn(OUT, IN)
    with pytest.raises(ValueError, match="3D"):
        opt.step()


# ── addressing: writing the mask, and stepping only part of the model ──

from moestack.config import OptimConfig                          # noqa: E402
from moestack.model import Transformer                           # noqa: E402

from conftest import tiny_args                                   # noqa: E402


def _built(**kw):
    torch.manual_seed(0)
    model = Transformer(tiny_args(**kw))
    return model, build_optimizers(model, OptimConfig(), split_experts=True)


def _expert_groups(opts):
    return [g for g in opts.param_groups
            if "expert" in g.get("roles", ()) and g.get("layer") is not None]


def test_set_expert_update_mask_puts_each_row_on_its_own_layer():
    model, opts = _built()
    n_layers, n_experts = len(model.layers), model.args.n_experts
    mask = torch.zeros(n_layers, n_experts, dtype=torch.bool)
    mask[0, 1] = mask[1, 2] = True
    opts.set_expert_update_mask(mask)
    for g in _expert_groups(opts):
        assert torch.equal(g["update_mask"], mask[g["layer"]])


def test_passing_none_clears_the_mask_back_to_the_unmasked_path():
    _, opts = _built()
    n_layers = len({g["layer"] for g in _expert_groups(opts)})
    opts.set_expert_update_mask(torch.ones(n_layers, 8, dtype=torch.bool))
    opts.set_expert_update_mask(None)
    assert all(g["update_mask"] is None for g in _expert_groups(opts))


def test_a_non_bool_mask_is_refused_before_it_reaches_the_optimizer():
    _, opts = _built()
    with pytest.raises(ValueError, match="bool tensor"):
        opts.set_expert_update_mask(torch.ones(2, 8))


def test_an_update_mask_under_adamw_raises_instead_of_silently_doing_nothing():
    torch.manual_seed(0)
    model = Transformer(tiny_args())
    opts = build_optimizers(model, OptimConfig(name="adamw"), split_experts=False)
    with pytest.raises(RuntimeError, match="per-expert update masks"):
        opts.set_expert_update_mask(torch.ones(2, 8, dtype=torch.bool))


def test_stepping_only_the_experts_leaves_the_rest_of_the_model_and_its_grads_alone():
    """A mid-window expert event must not disturb the shared parameters, and must
    not throw away the gradient they have accumulated so far either."""
    model, opts = _built()
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    expert_ids = {id(p) for g in _expert_groups(opts) for p in g["params"]}
    others = [(n, p, p.detach().clone(), p.grad.clone())
              for n, p in model.named_parameters() if id(p) not in expert_ids]
    experts = [(p, p.detach().clone()) for g in _expert_groups(opts) for p in g["params"]]

    opts.set_lr(0.02)
    opts.step(only="expert")

    for name, p, before, grad in others:
        assert torch.equal(p.data, before), f"{name} moved"
        assert p.grad is not None and torch.equal(p.grad, grad), f"{name} lost its gradient"
    for p, before in experts:
        assert not torch.equal(p.data, before)
