"""Expert-local gradient accumulation.

Stage 1: the mask inside Muon (moestack/optim/muon.py) and the addressing that
writes it (moestack/optim/builder.py). An expert that is still filling its
accumulation window must be left *completely* alone -- weights, decay and momentum
-- which is exactly what `lr_scale=0` does not give you.
"""

import math

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


# ──────────────────────────────────────────────────────────────────────
# Stage 2: the window controller (moestack/expert_accum.py)
# ──────────────────────────────────────────────────────────────────────

from moestack.expert_accum import (AccumState, accum_metrics, grad_scale,        # noqa: E402
                                   mean_tokens_per_expert, resolve_threshold)


def _state(*rows):
    s = AccumState.zeros(len(rows), len(rows[0]))
    return s


def _counts(*rows):
    return torch.tensor(rows, dtype=torch.long)


def _tick(state, counts, threshold=None, max_window=None, step=0):
    """One tick: observe, decide, close. Returns (event, tokens_at_event)."""
    state.observe(counts)
    ev = state.ready(threshold, max_window)
    return ev, state.consume(ev.fire, step)


# ── the threshold ────────────────────────────────────────────────────

def test_a_window_closes_when_it_reaches_the_threshold_not_after_it():
    s = AccumState.zeros(1, 2)
    ev, tokens = _tick(s, _counts([10, 9]), threshold=10)
    assert ev.fire.tolist() == [[True, False]]
    assert ev.by_threshold.tolist() == [[True, False]]
    assert tokens.tolist() == [[10, 0]]


def test_tokens_accumulate_across_ticks_until_the_budget_is_met():
    s = AccumState.zeros(1, 1)
    for _ in range(3):
        ev, _ = _tick(s, _counts([3]), threshold=10)
        assert not ev.any
    assert s.pending.tolist() == [[9]]
    ev, tokens = _tick(s, _counts([3]), threshold=10)
    assert ev.any and tokens.tolist() == [[12]]      # one tick can carry it past Q


def test_a_worked_example_interleaves_two_experts_on_their_own_clocks():
    """The point of the whole design in six lines: at a fixed token budget, two
    experts with different loads update on different ticks, and neither waits for
    the other."""
    s = AccumState.zeros(1, 2)
    fired = []
    for step, c in enumerate([[4, 12], [7, 3], [2, 8], [9, 1]]):
        ev, _ = _tick(s, _counts(c), threshold=10, step=step)
        fired.append(ev.fire[0].tolist())
    assert fired == [[False, True], [True, False], [False, True], [True, False]]
    assert s.events.tolist() == [[2, 2]]             # same budget, same event count


# ── the timeout ──────────────────────────────────────────────────────

def test_the_timeout_closes_a_window_that_never_fills():
    s = AccumState.zeros(1, 1)
    for _ in range(2):
        ev, _ = _tick(s, _counts([1]), threshold=100, max_window=3)
        assert not ev.any
    ev, tokens = _tick(s, _counts([1]), threshold=100, max_window=3)
    assert ev.fire.tolist() == [[True]] and tokens.tolist() == [[3]]
    assert ev.by_timeout.tolist() == [[True]] and ev.by_threshold.tolist() == [[False]]


def test_a_window_is_aged_from_its_first_token_not_from_the_last_event():
    """A late contribution starts its own window. Otherwise an expert that was
    quiet for D ticks would be fired by the timeout the instant one token arrived,
    on a window one tick old."""
    s = AccumState.zeros(1, 1)
    for _ in range(5):
        ev, _ = _tick(s, _counts([0]), threshold=100, max_window=3)
        assert not ev.any
    assert s.age.tolist() == [[0]]
    ev, _ = _tick(s, _counts([2]), threshold=100, max_window=3)
    assert not ev.any and s.age.tolist() == [[1]]


def test_an_expert_with_no_tokens_never_fires_however_long_it_waits():
    """The rule the mask in Muon exists to enforce: a zero gradient through
    Newton-Schulz is a full-magnitude step in a stale direction."""
    s = AccumState.zeros(1, 2)
    for _ in range(20):
        ev, _ = _tick(s, _counts([0, 5]), threshold=1, max_window=1)
        assert ev.fire[0, 0].item() is False
    assert s.events.tolist() == [[0, 20]]


def test_threshold_one_and_window_one_is_the_active_only_control():
    """Q=1, D=1 reduces to 'update exactly the experts that saw tokens this tick' --
    the control that separates not-stepping-dead-experts from accumulation itself."""
    s = AccumState.zeros(2, 3)
    c = _counts([5, 0, 1], [0, 0, 9])
    ev, tokens = _tick(s, c, threshold=1, max_window=1)
    assert torch.equal(ev.fire, c > 0)
    assert torch.equal(tokens, c)
    assert s.pending.sum() == 0


# ── bookkeeping ──────────────────────────────────────────────────────

def test_consume_resets_only_the_experts_that_fired():
    s = AccumState.zeros(1, 3)
    s.observe(_counts([10, 4, 0]))
    ev = s.ready(threshold=10, max_window=None)
    tokens = s.consume(ev.fire, step=7)
    assert tokens.tolist() == [[10, 0, 0]]
    assert s.pending.tolist() == [[0, 4, 0]]
    assert s.age.tolist() == [[0, 1, 0]]
    assert s.events.tolist() == [[1, 0, 0]]
    assert s.last_event.tolist() == [[7, -1, -1]]


def test_layers_are_independent():
    s = AccumState.zeros(2, 2)
    ev, _ = _tick(s, _counts([10, 0], [0, 10]), threshold=10)
    assert ev.fire.tolist() == [[True, False], [False, True]]


def test_flush_closes_every_open_window_and_nothing_else():
    s = AccumState.zeros(1, 3)
    s.observe(_counts([1, 0, 7]))
    ev = s.flush()
    assert ev.fire.tolist() == [[True, False, True]]
    tokens = s.consume(ev.fire, step=99)
    assert tokens.tolist() == [[1, 0, 7]] and s.pending.sum() == 0


def test_state_survives_a_checkpoint_round_trip():
    s = AccumState.zeros(2, 3)
    _tick(s, _counts([10, 4, 0], [1, 1, 1]), threshold=10, step=3)
    restored = AccumState.zeros(2, 3)
    restored.load_state_dict(s.state_dict())
    for k in ("pending", "age", "events", "last_event"):
        assert torch.equal(getattr(restored, k), getattr(s, k)), k


def test_the_same_counts_always_give_the_same_decisions():
    """Every rank runs this on identically reduced counts and must reach the same
    verdict: a replica that fires a different expert diverges silently."""
    a, b = AccumState.zeros(2, 4), AccumState.zeros(2, 4)
    g = torch.Generator().manual_seed(0)
    for step in range(10):
        c = torch.randint(0, 8, (2, 4), generator=g)
        ea, _ = _tick(a, c, threshold=12, max_window=4, step=step)
        eb, _ = _tick(b, c, threshold=12, max_window=4, step=step)
        assert torch.equal(ea.fire, eb.fire)
    assert torch.equal(a.events, b.events)


# ── misuse ───────────────────────────────────────────────────────────

def test_a_rule_with_neither_a_threshold_nor_a_window_is_an_error():
    s = AccumState.zeros(1, 2)
    s.observe(_counts([5, 5]))
    with pytest.raises(ValueError, match="no expert would ever update"):
        s.ready(None, None)


@pytest.mark.parametrize("call", [
    lambda s: s.observe(torch.zeros(3, 3, dtype=torch.long)),
    lambda s: s.consume(torch.zeros(3, 3, dtype=torch.bool), 0),
])
def test_a_shape_mismatch_is_an_error(call):
    with pytest.raises(ValueError, match="match the state's"):
        call(AccumState.zeros(1, 2))


# ── gradient scaling ─────────────────────────────────────────────────

def test_mean_scaling_divides_only_the_firing_experts_by_their_own_token_count():
    tokens = _counts([20, 0, 5])
    fire = torch.tensor([[True, False, True]])
    s = grad_scale(tokens, fire, "mean")
    torch.testing.assert_close(s, torch.tensor([[1 / 20, 1.0, 1 / 5]]))


def test_sum_scaling_leaves_every_gradient_alone():
    s = grad_scale(_counts([20, 0, 5]), torch.tensor([[True, False, True]]), "sum")
    assert (s == 1.0).all()


def test_an_unknown_grad_scale_is_an_error():
    with pytest.raises(ValueError, match="grad_scale must be one of"):
        grad_scale(_counts([1]), torch.tensor([[True]]), "median")


def test_mean_scaling_makes_two_windows_of_different_lengths_comparable():
    """Why the division exists: a long window and a short one must hand the
    momentum buffer gradients of the same magnitude, or window length alone decides
    which events dominate it."""
    long_g = torch.full((4, 3), 2.0) * 50      # 50 ticks of the same per-token grad
    short_g = torch.full((4, 3), 2.0) * 2      # 2 ticks of it
    sl = grad_scale(_counts([50]), torch.tensor([[True]]), "mean")[0, 0]
    ss = grad_scale(_counts([2]), torch.tensor([[True]]), "mean")[0, 0]
    torch.testing.assert_close(long_g * sl, short_g * ss)


# ── thresholds in real units ─────────────────────────────────────────

def test_a_balanced_expert_sees_tokens_times_topk_over_experts():
    assert mean_tokens_per_expert(16 * 2048, top_k=6, n_experts=64) == 3072.0


def test_a_relative_threshold_asks_for_that_many_ticks_of_average_load():
    q = resolve_threshold(None, 2.0, tokens_per_tick=16 * 2048, top_k=6, n_experts=64)
    assert q == 6144


def test_an_absolute_threshold_is_taken_as_written():
    assert resolve_threshold(1024, None, tokens_per_tick=1, top_k=1, n_experts=1) == 1024


@pytest.mark.parametrize("kw, match", [
    (dict(threshold=None, threshold_rel=None), "exactly one of"),
    (dict(threshold=1024, threshold_rel=2.0), "exactly one of"),
    (dict(threshold=0, threshold_rel=None), "at least 1 token"),
    (dict(threshold=None, threshold_rel=0.0), "must be positive"),
    (dict(threshold=None, threshold_rel=-1.0), "must be positive"),
])
def test_an_ill_defined_threshold_is_an_error(kw, match):
    with pytest.raises(ValueError, match=match):
        resolve_threshold(tokens_per_tick=1000, top_k=1, n_experts=10, **kw)


# ── metrics ──────────────────────────────────────────────────────────

def test_metrics_report_the_realised_budget_and_what_is_still_waiting():
    s = AccumState.zeros(1, 4)
    s.observe(_counts([12, 11, 3, 0]))
    ev = s.ready(threshold=10, max_window=None)
    tokens = s.consume(ev.fire, step=0)
    m = accum_metrics(s, ev, tokens, threshold=10)
    assert m["accum/event_fraction"] == 0.5
    assert m["accum/waiting_fraction"] == 0.25        # the expert sitting on 3
    assert m["accum/never_fired_fraction"] == 0.5
    assert m["accum/tokens_per_event_mean"] == 11.5
    assert m["accum/overshoot_mean"] == pytest.approx(1.15)
    assert m["accum/timeout_fraction"] == 0.0


def test_metrics_separate_a_timeout_from_a_met_budget():
    s = AccumState.zeros(1, 2)
    s.observe(_counts([50, 1]))
    s.age += 4
    ev = s.ready(threshold=10, max_window=4)
    m = accum_metrics(s, ev, s.consume(ev.fire, 0), threshold=10)
    assert m["accum/timeout_fraction"] == 0.5


def test_metrics_are_finite_when_nothing_fires():
    s = AccumState.zeros(1, 2)
    s.observe(_counts([1, 1]))
    ev = s.ready(threshold=10, max_window=None)
    m = accum_metrics(s, ev, s.consume(ev.fire, 0), threshold=10)
    assert m["accum/event_fraction"] == 0.0
    assert all(not math.isnan(v) for v in m.values())
    assert "accum/tokens_per_event_mean" not in m
