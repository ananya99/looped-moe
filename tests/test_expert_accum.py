"""Expert-local gradient accumulation.

Stage 1: the mask inside Muon (moestack/optim/muon.py) and the addressing that
writes it (moestack/optim/builder.py). An expert that is still filling its
accumulation window must be left *completely* alone -- weights, decay and momentum
-- which is exactly what `lr_scale=0` does not give you.
"""

import math
from pathlib import Path

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

from moestack.expert_accum import (AccumState, EventTally, accum_metrics,        # noqa: E402
                                   grad_scale, mean_tokens_per_expert,
                                   resolve_threshold)


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


def test_no_threshold_at_all_is_the_fixed_window_variant():
    """Neither setting means the window is closed by max_window alone -- a real
    configuration (fixed-length windows), not a mistake. `ready` is what refuses a
    rule with no closing condition at all."""
    assert resolve_threshold(None, None, tokens_per_tick=1000, top_k=1,
                             n_experts=10) is None


@pytest.mark.parametrize("kw, match", [
    (dict(threshold=1024, threshold_rel=2.0), "at most one of"),
    (dict(threshold=0, threshold_rel=None), "at least 1 token"),
    (dict(threshold=None, threshold_rel=0.0), "must be positive"),
    (dict(threshold=None, threshold_rel=-1.0), "must be positive"),
])
def test_an_ill_defined_threshold_is_an_error(kw, match):
    with pytest.raises(ValueError, match=match):
        resolve_threshold(tokens_per_tick=1000, top_k=1, n_experts=10, **kw)


# ── metrics ──────────────────────────────────────────────────────────

def _tally(state, event, tokens):
    t = EventTally(state.pending.numel())
    t.add(event, tokens)
    return t


def test_metrics_report_the_realised_budget_and_what_is_still_waiting():
    s = AccumState.zeros(1, 4)
    s.observe(_counts([12, 11, 3, 0]))
    ev = s.ready(threshold=10, max_window=None)
    tokens = s.consume(ev.fire, step=0)
    m = accum_metrics(s, _tally(s, ev, tokens), threshold=10)
    assert m["accum/events_per_expert"] == 0.5
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
    m = accum_metrics(s, _tally(s, ev, s.consume(ev.fire, 0)), threshold=10)
    assert m["accum/timeout_fraction"] == 0.5


def test_metrics_are_finite_when_nothing_fires():
    s = AccumState.zeros(1, 2)
    s.observe(_counts([1, 1]))
    ev = s.ready(threshold=10, max_window=None)
    m = accum_metrics(s, _tally(s, ev, s.consume(ev.fire, 0)), threshold=10)
    assert m["accum/events_per_expert"] == 0.0
    assert all(not math.isnan(v) for v in m.values())
    assert "accum/tokens_per_event_mean" not in m


# ──────────────────────────────────────────────────────────────────────
# Stage 3: the trainer's seams (moestack/trainer.py, moestack/callbacks.py)
# ──────────────────────────────────────────────────────────────────────

from moestack.callbacks import Callback, StepState               # noqa: E402
from moestack.trainer import Trainer                             # noqa: E402

from conftest import tiny_config                                 # noqa: E402


class _Recorder(Callback):
    """Writes down every event it is handed, in order."""

    def __init__(self, flush_moved=False):
        self.log, self.flush_moved = [], flush_moved

    def on_microbatch_end(self, trainer, state, micro_step, counts, loss):
        self.log.append(("micro", state.step, micro_step, tuple(counts.shape)))

    def on_before_optimizer_step(self, trainer, state):
        self.log.append(("before", state.step))

    def on_accepted_step(self, trainer, state):
        self.log.append(("accepted", state.step))

    def on_step_end(self, trainer, state):
        self.log.append(("end", state.step))

    def on_eval(self, trainer, state, metrics):
        self.log.append(("eval", trainer.step))

    def on_before_final_eval(self, trainer, state):
        self.log.append(("flush", trainer.step))
        return self.flush_moved

    def kinds(self, step):
        return [e[0] for e in self.log if len(e) > 1 and e[1] == step and e[0] != "eval"]


def _fit(tmp_path, cb, **sections):
    cfg = tiny_config(tmp_path, **sections)
    t = Trainer(cfg, callbacks=[cb], verbose=False)
    t.fit()
    return t


def test_a_microbatch_event_fires_once_per_microbatch_before_the_step_events(tmp_path):
    cb = _Recorder()
    t = _fit(tmp_path, cb, train=dict(n_steps=2))
    accum = t.cfg.data.grad_accum
    assert cb.kinds(0) == ["micro"] * accum + ["before", "accepted", "end"]
    assert cb.kinds(1) == ["micro"] * accum + ["before", "accepted", "end"]
    assert [e[2] for e in cb.log if e[0] == "micro" and e[1] == 0] == list(range(accum))


def test_the_microbatch_event_carries_that_microbatchs_own_counts(tmp_path):
    seen = []

    class Watch(Callback):
        def on_microbatch_end(self, trainer, state, micro_step, counts, loss):
            seen.append(counts.clone())

        def on_step_end(self, trainer, state):
            seen.append(state.expert_counts.clone())

    t = _fit(tmp_path, Watch(), train=dict(n_steps=1))
    *micro, total = seen
    assert len(micro) == t.cfg.data.grad_accum
    # The step's reduced counts are the micro-batches summed (dp=1 in this test).
    assert torch.equal(torch.stack(micro).sum(0), total)
    assert all(c.shape == total.shape for c in micro)


class _Poisoner(Callback):
    """Makes one step's gradient non-finite, the way a real loss spike would."""

    def __init__(self, at_step):
        self.at_step = at_step

    def on_microbatch_end(self, trainer, state, micro_step, counts, loss):
        if state.step == self.at_step and micro_step == trainer.cfg.data.grad_accum - 1:
            next(iter(trainer.model.parameters())).grad.fill_(float("inf"))


def test_a_dropped_step_reaches_on_before_optimizer_step_but_not_on_accepted_step(tmp_path):
    """The reason `on_accepted_step` exists. A rejected batch must leave the
    accumulator exactly as it found it, and `on_before_optimizer_step` fires before
    the trainer knows whether the batch is any good."""
    cb = _Recorder()
    cfg = tiny_config(tmp_path, train=dict(n_steps=3))
    t = Trainer(cfg, callbacks=[cb, _Poisoner(at_step=1)], verbose=False)
    t.fit()
    assert "before" in cb.kinds(1) and "accepted" not in cb.kinds(1)
    assert "accepted" in cb.kinds(0) and "accepted" in cb.kinds(2)
    assert t.skipped == 1


def test_the_final_flush_fires_once_on_a_normal_finish(tmp_path):
    cb = _Recorder()
    _fit(tmp_path, cb, train=dict(n_steps=2))
    assert [e[0] for e in cb.log].count("flush") == 1


def test_a_flush_that_moves_the_weights_forces_the_final_eval_to_be_redone(tmp_path):
    """Otherwise the number the run reports describes the model one update ago."""
    quiet, moved = _Recorder(flush_moved=False), _Recorder(flush_moved=True)
    _fit(tmp_path / "a", quiet, train=dict(n_steps=4, eval_every=2))
    _fit(tmp_path / "b", moved, train=dict(n_steps=4, eval_every=2))
    assert [e[0] for e in moved.log].count("eval") == \
        [e[0] for e in quiet.log].count("eval") + 1


# ── gradients that are accumulators, not this batch's gradient ───────

class _Registers(Callback):
    """Registers one expert tensor as an accumulator and shouts into its gradient."""

    def __init__(self, register=True, spike=1e6):
        self.register, self.spike = register, spike
        self.p, self.seen = None, []

    def on_train_start(self, trainer):
        self.p = next(iter(trainer.model.moe_layers())).experts.gate_up_proj
        if self.register:
            trainer.grad_accumulators.append(self.p)

    def on_microbatch_end(self, trainer, state, micro_step, counts, loss):
        if micro_step == trainer.cfg.data.grad_accum - 1:
            self.p.grad.add_(self.spike)

    def on_accepted_step(self, trainer, state):
        self.seen.append((state.grad_norm, float(self.p.grad.abs().max())))


def test_a_registered_accumulator_is_left_out_of_the_gradient_clip(tmp_path):
    """Clipping rescales every gradient it is given. On an accumulator that is not
    clipping a gradient -- it is shrinking a half-filled window, by a different
    factor each step."""
    kept = _Registers(register=True)
    _fit(tmp_path / "kept", kept, train=dict(n_steps=1))
    norm, biggest = kept.seen[0]
    assert math.isfinite(norm) and norm < 1e4        # the spike is not in the norm
    assert biggest >= 1e6                            # ... and was not scaled away

    clipped = _Registers(register=False)
    _fit(tmp_path / "clipped", clipped, train=dict(n_steps=1))
    norm, biggest = clipped.seen[0]
    assert norm > 1e4 and biggest < 1e6              # the baseline does both


def test_a_registered_accumulator_survives_the_per_step_clear(tmp_path):
    """The central mechanic: `p.grad` is the window, so the trainer must not zero
    it between batches the way it zeroes everything else."""
    cb = _Registers(register=True, spike=1.0)
    _fit(tmp_path, cb, train=dict(n_steps=3))
    maxima = [b for _, b in cb.seen]
    assert len(maxima) == 3
    assert maxima[1] > maxima[0] and maxima[2] > maxima[1]


def test_nothing_is_registered_by_default_and_the_whole_model_is_clipped(tmp_path):
    t = _fit(tmp_path, _Recorder(), train=dict(n_steps=1))
    assert t.grad_accumulators == []
    assert list(t._clipped_params()) == list(t.model.parameters())


# ──────────────────────────────────────────────────────────────────────
# Stage 4: the callback on the step clock (moestack/callbacks.py)
# ──────────────────────────────────────────────────────────────────────

from moestack.callbacks import ExpertAccumulation                # noqa: E402
from moestack.config import Config                               # noqa: E402


def _accum_cfg(tmp_path, steps=3, **accum):
    cfg = tiny_config(tmp_path, train=dict(n_steps=steps))
    cfg.optim.expert_accum.enabled = True
    for k, v in accum.items():
        setattr(cfg.optim.expert_accum, k, v)
    cfg.__post_init__()
    return cfg


def _accum_fit(tmp_path, cb=None, steps=3, **accum):
    t = Trainer(_accum_cfg(tmp_path, steps, **accum), callbacks=[cb] if cb else [],
                verbose=False)
    t.fit()
    return t


def _expert_weights(trainer):
    return torch.cat([m.experts.gate_up_proj.detach().flatten()
                      for m in trainer.model.moe_layers()])


def _expert_grad_mass(trainer):
    """|grad| summed over each expert's gate_up matrix -> [L, E]."""
    return torch.stack([m.experts.gate_up_proj.grad.flatten(1).abs().sum(1)
                        for m in trainer.model.moe_layers()])


class _Watch(Callback):
    """Snapshots the mask the accumulator armed, and the state it left behind."""

    def __init__(self):
        self.rows, self.mask, self.start, self.shared0 = [], None, None, None

    def on_train_start(self, trainer):
        self.start = _expert_weights(trainer).clone()
        self.shared0 = trainer.model.tok_embeddings.weight.detach().clone()

    def on_accepted_step(self, trainer, state):
        groups = trainer.optimizers.expert_lr_groups()
        self.mask = torch.stack([g[0]["update_mask"]
                                 for _, g in sorted(groups.items())]).clone()

    def on_step_end(self, trainer, state):
        self.rows.append((self.mask, _expert_grad_mass(trainer).clone(),
                          state.expert_counts.clone(),
                          _expert_weights(trainer).clone()))


# ── the window, end to end ───────────────────────────────────────────

def test_experts_stay_frozen_while_their_windows_fill_and_the_rest_of_the_model_trains(tmp_path):
    w = _Watch()
    t = _accum_fit(tmp_path, w, steps=3, threshold=10**9)      # a budget nothing reaches
    assert all(not m.any() for m, *_ in w.rows)                # nobody ever fired
    for _, _, _, weights in w.rows:
        assert torch.equal(weights, w.start)                   # ... and nobody moved
    # The shared parameters trained all along: this is not a frozen model.
    assert not torch.equal(t.model.tok_embeddings.weight.detach(), w.shared0)
    assert t.step == 3


def test_a_waiting_experts_gradient_keeps_growing_instead_of_being_cleared(tmp_path):
    w = _Watch()
    _accum_fit(tmp_path, w, steps=3, threshold=10**9)
    mass = [g.sum().item() for _, g, _, _ in w.rows]
    assert mass[1] > mass[0] and mass[2] > mass[1]


def test_the_final_flush_applies_the_window_the_run_never_filled(tmp_path):
    """Without it, every expert's last partial window is silently thrown away --
    and with a budget this large, that is the entire run."""
    w = _Watch()
    t = _accum_fit(tmp_path, w, steps=3, threshold=10**9)
    assert not torch.equal(_expert_weights(t), w.start)


def test_at_a_budget_of_one_token_exactly_the_experts_that_saw_tokens_fire(tmp_path):
    """Q=1, D=1 is the active-only control: no accumulation, but no stepping a
    zero-token expert on renormalised stale momentum either."""
    w = _Watch()
    _accum_fit(tmp_path, w, steps=2, threshold=1, max_window=1)
    for mask, _, counts, _ in w.rows:
        assert torch.equal(mask, counts > 0)


def test_a_fired_window_is_cleared_and_the_others_are_left_filling(tmp_path):
    w = _Watch()
    _accum_fit(tmp_path, w, steps=4, threshold=100)
    fired_any = False
    for mask, mass, _, _ in w.rows:
        assert mass[mask].sum() == 0, "a fired expert kept its gradient"
        fired_any |= bool(mask.any())
        if (~mask).any():
            assert mass[~mask].sum() > 0, "a waiting expert lost its gradient"
    assert fired_any, "nothing fired, so this proves nothing -- lower the threshold"


def test_an_expert_that_waits_takes_exactly_the_step_the_summed_gradient_would(tmp_path):
    """The arithmetic the whole scheme rests on, at the optimizer level: three
    batches accumulated into `p.grad` behind a closed mask, then fired, equal one
    step on their sum."""
    g1, g2, g3 = _grads(3, seed=11)
    w0 = torch.randn(E, OUT, IN)

    p = torch.nn.Parameter(w0.clone())
    opt = Muon([p], **MUON_KW)
    p.grad = torch.zeros_like(p)
    for i, g in enumerate((g1, g2, g3)):
        p.grad += g                                    # never cleared
        opt.param_groups[0]["update_mask"] = torch.full((E,), i == 2, dtype=torch.bool)
        opt.step()

    summed, _ = _run(torch.nn.Parameter(w0.clone()), [g1 + g2 + g3])
    torch.testing.assert_close(p.data, summed.data)


# ── a batch the trainer rejects ──────────────────────────────────────

def test_a_dropped_batch_throws_away_every_open_window(tmp_path):
    """Its gradient is already in the accumulators and cannot be subtracted back
    out, so carrying on would mean every expert's next update contains it."""
    cfg = _accum_cfg(tmp_path, steps=3, threshold=10**9)
    w = _Watch()
    t = Trainer(cfg, callbacks=[_Poisoner(at_step=1), w], verbose=False)
    t.fit()
    acc = next(c for c in t.callbacks.callbacks if isinstance(c, ExpertAccumulation))
    assert acc.discarded == 1
    # step 0 filled a window, step 1 was dropped and cleared it, step 2 refilled.
    mass = [g.sum().item() for _, g, _, _ in w.rows]
    assert mass[1] == 0 and 0 < mass[2] < mass[0] * 2


# ── configuration ────────────────────────────────────────────────────

def test_accumulation_is_off_by_default_and_changes_no_run_name():
    cfg = Config()
    assert not cfg.optim.expert_accum.enabled
    assert "acc" not in cfg.run_name().split("-lr")[1]


def test_the_trainer_only_builds_the_callback_when_it_is_enabled(tmp_path):
    plain = Trainer(tiny_config(tmp_path, train=dict(n_steps=1)), verbose=False)
    assert not any(isinstance(c, ExpertAccumulation) for c in plain.callbacks.callbacks)
    on = Trainer(_accum_cfg(tmp_path, steps=1, threshold=100), verbose=False)
    assert any(isinstance(c, ExpertAccumulation) for c in on.callbacks.callbacks)


def test_enabling_it_splits_the_expert_tensors_so_they_can_be_addressed(tmp_path):
    t = Trainer(_accum_cfg(tmp_path, steps=1, threshold=100), verbose=False)
    assert t.optimizers.expert_lr_groups()


def test_every_cell_of_the_grid_gets_its_own_name(tmp_path):
    names = set()
    for kw in (dict(threshold_rel=2.0, max_window=8), dict(threshold_rel=4.0, max_window=8),
               dict(threshold_rel=2.0, max_window=16), dict(threshold=1, max_window=1),
               dict(threshold_rel=2.0, max_window=8, grad_scale="sum"),
               dict(max_window=4)):
        names.add(_accum_cfg(tmp_path, **kw).run_name())
    assert len(names) == 6


@pytest.mark.parametrize("kw, match", [
    (dict(threshold=100), "needs optim.name=muon"),
])
def test_accumulation_under_adamw_fails_at_startup(kw, match):
    cfg = Config()
    cfg.optim.name = "adamw"
    cfg.optim.expert_accum.enabled = True
    for k, v in kw.items():
        setattr(cfg.optim.expert_accum, k, v)
    with pytest.raises(ValueError, match=match):
        cfg.__post_init__()


@pytest.mark.parametrize("accum, train, match", [
    (dict(), {}, "no expert would ever update"),
    (dict(threshold=1, threshold_rel=1.0), {}, "at most one of"),
    (dict(threshold=1, clock="hourly"), {}, "clock must be"),
    (dict(threshold=1, grad_scale="median"), {}, "grad_scale must be"),
    (dict(threshold=1, max_window=0), {}, "at least 1 tick"),
    (dict(threshold=1), dict(auto_resume=True), "not resumable"),
    (dict(threshold=1), dict(resume="x.pt"), "not resumable"),
])
def test_an_ill_defined_accumulation_run_fails_at_startup(accum, train, match):
    cfg = Config()
    cfg.optim.expert_accum.enabled = True
    for k, v in accum.items():
        setattr(cfg.optim.expert_accum, k, v)
    for k, v in train.items():
        setattr(cfg.train, k, v)
    with pytest.raises(ValueError, match=match):
        cfg.__post_init__()


def test_accumulation_and_per_expert_lr_cannot_both_be_on():
    cfg = Config()
    cfg.optim.expert_accum.enabled = True
    cfg.optim.expert_accum.threshold = 100
    cfg.optim.per_expert_lr.alpha = 0.5
    with pytest.raises(ValueError, match="cannot both be on"):
        cfg.__post_init__()


def test_counting_several_events_for_one_expert_keeps_the_rate_above_one():
    """What tells the two clocks apart: on the micro-batch clock an expert can fire
    more than once in a step, and `events_per_expert` is where that shows up."""
    s = AccumState.zeros(1, 2)
    t = EventTally(2)
    for _ in range(3):
        s.observe(_counts([10, 0]))
        ev = s.ready(threshold=10, max_window=None)
        t.add(ev, s.consume(ev.fire, 0))
    m = accum_metrics(s, t, threshold=10)
    assert m["accum/events_per_expert"] == 1.5           # 3 events over 2 experts
    assert m["accum/tokens_per_event_mean"] == 10.0


# ──────────────────────────────────────────────────────────────────────
# Stage 5: the micro-batch clock -- experts stop waiting for the batch
# ──────────────────────────────────────────────────────────────────────

def test_the_microbatch_clock_lets_an_expert_update_several_times_in_one_batch(tmp_path):
    """The whole difference from the step clock. At a budget an average expert
    meets every micro-batch, `events_per_expert` climbs towards grad_accum instead
    of sitting at the baseline's 1.0."""
    seen = []

    class Rate(Callback):
        def on_step_end(self, trainer, state):
            seen.append(state.metrics["accum/events_per_expert"])

    t = _accum_fit(tmp_path, Rate(), steps=3, threshold=1, max_window=1,
                   clock="microbatch")
    assert max(seen) > 1.0
    assert max(seen) <= t.cfg.data.grad_accum


def test_the_two_clocks_agree_when_a_batch_is_a_single_microbatch(tmp_path):
    """With grad_accum=1 a tick is a tick either way, so the clocks are the same
    experiment -- and must produce the same weights."""
    out = {}
    for clock in ("step", "microbatch"):
        cfg = _accum_cfg(tmp_path / clock, steps=4, threshold=50, max_window=4,
                         clock=clock)
        cfg.data.grad_accum = 1
        t = Trainer(cfg, verbose=False)
        t.fit()
        out[clock] = _expert_weights(t)
    torch.testing.assert_close(out["step"], out["microbatch"], atol=0, rtol=0)


def test_the_batch_step_does_not_step_the_experts_a_second_time(tmp_path):
    """Everything an expert does on this clock happens inside the loop. Without an
    all-off mask at the step, Muon would step whatever fired last a second time --
    on its freshly zeroed gradient, i.e. on renormalised stale momentum."""
    seen = []

    class MaskAtStep(Callback):
        def on_accepted_step(self, trainer, state):
            groups = trainer.optimizers.expert_lr_groups()
            seen.append(torch.stack([g[0]["update_mask"]
                                     for _, g in sorted(groups.items())]).clone())

    _accum_fit(tmp_path, MaskAtStep(), steps=3, threshold=1, max_window=1,
               clock="microbatch")
    assert seen and all(not m.any() for m in seen)


def test_a_waiting_expert_is_still_untouched_on_the_microbatch_clock(tmp_path):
    w = _Watch()
    t = _accum_fit(tmp_path, w, steps=3, threshold=10**9, clock="microbatch")
    for _, _, _, weights in w.rows:
        assert torch.equal(weights, w.start)
    assert not torch.equal(_expert_weights(t), w.start)      # ... until the flush


def test_a_nonfinite_microbatch_fires_nothing(tmp_path):
    """An expert that has stepped cannot be un-stepped, so a bad micro-batch must
    not be allowed to close a window. The batch is dropped at the step boundary
    and the windows are discarded there instead."""
    cfg = _accum_cfg(tmp_path, steps=1, threshold=1, max_window=1, clock="microbatch")
    t = Trainer(cfg, verbose=False)
    t.callbacks.on_train_start(t)
    acc = next(c for c in t.callbacks.callbacks if isinstance(c, ExpertAccumulation))
    t.train_step()                                   # real grads, real events

    before = acc.state.events.clone()
    counts = torch.ones_like(acc.state.pending)
    acc.on_microbatch_end(t, StepState(step=1), 0, counts, torch.tensor(float("inf")))
    assert acc.nonfinite_micro == 1
    assert torch.equal(acc.state.events, before)     # nothing fired
    assert acc.state.pending.sum() == 0              # ... and nothing was observed


def test_the_microbatch_clock_refuses_to_run_under_ddp(tmp_path):
    """Mid-batch an expert's gradient is this rank's alone (the trainer runs all
    but the last micro-batch under no_sync), so the replicas would silently take
    different steps."""
    t = Trainer(_accum_cfg(tmp_path, steps=1, threshold=100, clock="microbatch"),
                verbose=False)
    t.dist.world_size = 2
    with pytest.raises(NotImplementedError, match="single-process for now"):
        t.callbacks.on_train_start(t)


def test_the_step_clock_is_still_allowed_under_ddp(tmp_path):
    t = Trainer(_accum_cfg(tmp_path, steps=1, threshold=100, clock="step"), verbose=False)
    t.dist.world_size = 2
    t.callbacks.on_train_start(t)                    # does not raise


def test_the_microbatch_clocks_budget_is_read_per_microbatch(tmp_path):
    """threshold_rel means 'this many ticks of a balanced expert's load', and a
    tick is a micro-batch here -- so the same setting is grad_accum times smaller
    than on the step clock. The two cells are only comparable at a fixed Q."""
    per_clock = {}
    for clock in ("step", "microbatch"):
        t = Trainer(_accum_cfg(tmp_path / clock, steps=1, threshold_rel=2.0, clock=clock),
                    verbose=False)
        t.callbacks.on_train_start(t)
        acc = next(c for c in t.callbacks.callbacks if isinstance(c, ExpertAccumulation))
        per_clock[clock] = acc.threshold
    assert per_clock["step"] == per_clock["microbatch"] * 2      # tiny_config: accum=2


def test_the_two_clocks_get_different_names():
    a = _accum_cfg(Path("/tmp"), threshold=512, max_window=8, clock="step").run_name()
    b = _accum_cfg(Path("/tmp"), threshold=512, max_window=8, clock="microbatch").run_name()
    assert "accst" in a and "accmb" in b and a != b


# ──────────────────────────────────────────────────────────────────────
# Stage 6: the record of what happened
# ──────────────────────────────────────────────────────────────────────

def test_update_counts_have_their_own_inequality_measure():
    """load/gini says the tokens are unevenly spread. This says the *updates* are
    -- which on the micro-batch clock can run away even while routing does not."""
    even, lopsided = AccumState.zeros(1, 4), AccumState.zeros(1, 4)
    even.events += torch.tensor([[5, 5, 5, 5]])
    lopsided.events += torch.tensor([[20, 1, 1, 1]])
    t = EventTally(4)
    assert accum_metrics(even, t, None)["accum/event_gini"] == 0.0
    assert accum_metrics(lopsided, t, None)["accum/event_gini"] > 0.4


def _events_file(trainer):
    return Path(trainer.save_dir) / f"{trainer.run_name}_accum_events.jsonl"


def _records(trainer):
    import json
    return [json.loads(line) for line in _events_file(trainer).read_text().splitlines()]


def test_the_events_file_records_every_tick_that_fired(tmp_path):
    cfg = _accum_cfg(tmp_path, steps=4, threshold=1, max_window=1)
    cfg.logging.accum_events_every = 1
    t = Trainer(cfg, verbose=False)
    t.fit()
    rows = _records(t)
    assert rows, "nothing was written"
    r = rows[0]
    L, E = t.args.n_layers, t.args.n_experts
    assert len(r["tokens"]) == L and len(r["tokens"][0]) == E
    assert set(r) == {"step", "tick", "clock", "reason", "threshold", "tokens",
                      "by_timeout", "pending", "age", "events"}
    # tokens is zero wherever nothing fired, so it carries the mask too
    assert any(v > 0 for row in r["tokens"] for v in row)


def test_the_cadence_is_respected_and_zero_writes_nothing(tmp_path):
    cfg = _accum_cfg(tmp_path / "every2", steps=6, threshold=1, max_window=1)
    cfg.logging.accum_events_every = 2
    t = Trainer(cfg, verbose=False)
    t.fit()
    assert {r["step"] for r in _records(t)} <= {0, 2, 4}

    off = Trainer(_accum_cfg(tmp_path / "off", steps=4, threshold=1, max_window=1),
                  verbose=False)
    off.fit()
    assert not _events_file(off).exists()


def test_the_microbatch_clock_records_one_row_per_firing_microbatch(tmp_path):
    cfg = _accum_cfg(tmp_path, steps=2, threshold=1, max_window=1, clock="microbatch")
    cfg.logging.accum_events_every = 1
    t = Trainer(cfg, verbose=False)
    t.fit()
    ticks = [r["tick"] for r in _records(t) if r["step"] == 0 and r["reason"] == "rule"]
    assert ticks == sorted(ticks) and len(ticks) > 1
    assert max(ticks) < t.cfg.data.grad_accum


def test_the_final_flush_is_recorded_and_says_so(tmp_path):
    cfg = _accum_cfg(tmp_path, steps=2, threshold=10**9)
    cfg.logging.accum_events_every = 1
    t = Trainer(cfg, verbose=False)
    t.fit()
    rows = _records(t)
    assert [r["reason"] for r in rows] == ["flush"]       # nothing else ever fired
    assert rows[0]["step"] == 2


def test_the_events_cadence_does_not_change_a_run_name(tmp_path):
    """It is measurement, so it lives in `logging` and must stay out of the
    fingerprint -- two runs that differ only in it are the same experiment."""
    a = _accum_cfg(tmp_path, threshold=100)
    b = _accum_cfg(tmp_path, threshold=100)
    b.logging.accum_events_every = 25
    assert a.run_name() == b.run_name()
