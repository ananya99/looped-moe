"""Per-expert gradient quality (moestack/grad_stats.py) and the clock that drives
it (callbacks.ExpertGradStats).

The claim under test is the thesis's: an expert that sees few tokens gets an update
assembled from few, disagreeing micro-gradients, and widening its window raises the
signal-to-noise ratio. These pin the arithmetic on CPU, plus the two things that
would silently corrupt it -- a hook firing twice per backward, and a window measured
against micro-batches that did not feed it.
"""
import math

import pytest
import torch
import torch.nn as nn

from moestack.grad_stats import ExpertGradMoments, accumulated_grad_sq


class _Experts(nn.Module):
    """The seam `grad_stats` hooks: experts as single 3D [E, ...] tensors."""

    def __init__(self, E=4, D=3):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(E, D, D))
        self.down_proj = nn.Parameter(torch.zeros(E, D, D))


class _Layer(nn.Module):
    def __init__(self, E=4, D=3):
        super().__init__()
        self.experts = _Experts(E, D)


def _layers(n=2, E=4, D=3):
    return [_Layer(E, D) for _ in range(n)]


def _backward(layers, grads):
    """One micro-batch: push `grads[li]` ([E, D, D]) into each layer's two matrices."""
    for layer, g in zip(layers, grads):
        for p in (layer.experts.gate_up_proj, layer.experts.down_proj):
            p.backward(gradient=g, retain_graph=True) if p.grad_fn else None
    # leaf tensors have no grad_fn; drive the hooks through autograd instead
    for layer, g in zip(layers, grads):
        for p in (layer.experts.gate_up_proj, layer.experts.down_proj):
            torch.autograd.backward(p * 1.0, g, retain_graph=False)


# ── the identity ─────────────────────────────────────────────────────

def test_identical_micro_gradients_have_no_noise_to_measure():
    """Every micro-batch agreeing exactly means zero variance, so SNR is not a
    number -- NaN, not a huge value and not zero."""
    layers = _layers(n=1, E=1, D=2)
    m = ExpertGradMoments(1, 1).attach(layers)
    m.arm()
    g = torch.ones(1, 2, 2)
    for _ in range(4):
        _backward(layers, [g])
        m.tick()
    out = m.close(accumulated_grad_sq(layers))
    assert math.isnan(float(out["snr"][0, 0]))


def test_snr_rises_as_the_window_widens():
    """The thesis prediction: averaging independent noise over more micro-batches
    makes the mean direction stand out, roughly as sqrt(M)."""
    torch.manual_seed(0)
    signal = torch.ones(1, 4, 4) * 0.5

    def snr_over(M):
        layers = _layers(n=1, E=1, D=4)
        m = ExpertGradMoments(1, 1).attach(layers)
        m.arm()
        for _ in range(M):
            _backward(layers, [signal + torch.randn(1, 4, 4)])
            m.tick()
        return float(m.close(accumulated_grad_sq(layers))["snr"][0, 0])

    few, many = snr_over(4), snr_over(64)
    assert many > few > 0
    # 16x the window should buy ~sqrt(16) = 4x the SNR; loosely bounded because
    # this is one noise draw, not an expectation.
    assert 2.0 < many / few < 8.0


def test_an_expert_that_fires_once_sits_at_the_floor():
    """One non-zero micro-batch out of M gives exactly sqrt(M/(M-1)) -- about 1.0,
    whatever M is. A single sample's mean is worth about as much as its own
    uncertainty, and lengthening the window around it must not change that."""
    M = 5
    layers = _layers(n=1, E=1, D=2)
    m = ExpertGradMoments(1, 1).attach(layers)
    m.arm()
    for i in range(M):
        _backward(layers, [torch.ones(1, 2, 2) if i == 0 else torch.zeros(1, 2, 2)])
        m.tick()
    snr = float(m.close(accumulated_grad_sq(layers))["snr"][0, 0])
    assert snr == pytest.approx(math.sqrt(M / (M - 1)), rel=1e-4)


def test_an_expert_that_never_fires_is_nan_not_zero():
    layers = _layers(n=1, E=2, D=2)
    m = ExpertGradMoments(1, 2).attach(layers)
    m.arm()
    g = torch.zeros(2, 2, 2)
    g[0] = 1.0                                   # expert 1 stays dark
    for _ in range(4):
        _backward(layers, [g])
        m.tick()
    out = m.close(accumulated_grad_sq(layers))
    assert math.isnan(float(out["snr"][0, 1]))


def test_snr_is_invariant_to_gradient_scale():
    """It is a ratio, so the (loss / accum) scaling and Muon's normalisation cannot
    move it. If this breaks, something is mixing scaled and unscaled quantities."""
    torch.manual_seed(1)
    draws = [torch.randn(1, 3, 3) for _ in range(6)]

    def snr_scaled(k):
        layers = _layers(n=1, E=1, D=3)
        m = ExpertGradMoments(1, 1).attach(layers)
        m.arm()
        for g in draws:
            _backward(layers, [g * k])
            m.tick()
        return float(m.close(accumulated_grad_sq(layers))["snr"][0, 0])

    assert snr_scaled(1.0) == pytest.approx(snr_scaled(100.0), rel=1e-3)


def test_both_expert_matrices_enter_the_same_window():
    """||g||^2 = ||g_gate||^2 + ||g_down||^2, which is what lets the two hooks
    accumulate into one array without being paired per micro-batch."""
    layers = _layers(n=1, E=1, D=2)
    m = ExpertGradMoments(1, 1).attach(layers)
    m.arm()
    _backward(layers, [torch.ones(1, 2, 2)])
    # 2 matrices x 4 entries of 1.0
    assert float(m.sumsq[0, 0]) == pytest.approx(8.0)


# ── the hazards ──────────────────────────────────────────────────────

def test_a_disarmed_window_sees_nothing():
    """Measurement runs on a cadence, so the hooks must be inert off-cadence --
    otherwise an unmeasured step still pays for them and pollutes the next window."""
    layers = _layers(n=1, E=1, D=2)
    m = ExpertGradMoments(1, 1).attach(layers)
    m.disarm()
    _backward(layers, [torch.ones(1, 2, 2)])
    assert float(m.sumsq[0, 0]) == 0.0


def test_reset_clears_only_the_experts_that_fired():
    """Under accumulation two experts in a layer are at different points in their
    windows; resetting both would measure the survivor against a window it did not
    have."""
    layers = _layers(n=1, E=2, D=2)
    m = ExpertGradMoments(1, 2).attach(layers)
    m.arm()
    _backward(layers, [torch.ones(2, 2, 2)])
    m.tick()
    fire = torch.tensor([[True, False]])
    m.reset(fire)
    assert float(m.sumsq[0, 0]) == 0.0 and float(m.sumsq[0, 1]) > 0.0
    assert m.micro.tolist() == [[0, 1]]


def test_detach_removes_the_hooks():
    layers = _layers(n=1, E=1, D=2)
    m = ExpertGradMoments(1, 1).attach(layers)
    m.arm()
    m.detach()
    _backward(layers, [torch.ones(1, 2, 2)])
    assert float(m.sumsq[0, 0]) == 0.0


def test_hooks_fire_once_per_backward_under_activation_checkpointing():
    """Non-reentrant checkpointing replays the forward inside backward. Forward
    hooks double-count under it -- that is why the router counts come from
    RouterOut. Parameter grad hooks must not, or every second moment is inflated
    and SNR reads low for the whole run."""
    from torch.utils.checkpoint import checkpoint

    E, D = 2, 3
    layer = _Layer(E, D)
    layer.experts.gate_up_proj.data.fill_(0.1)
    layer.experts.down_proj.data.fill_(0.1)
    m = ExpertGradMoments(1, E).attach([layer])

    def block(x):
        h = torch.einsum("ed,eod->eo", x, layer.experts.gate_up_proj)
        return torch.einsum("eo,edo->ed", h, layer.experts.down_proj)

    x = torch.ones(E, D, requires_grad=True)
    plain = block(x).sum()
    m.arm()
    plain.backward()
    direct = m.sumsq.clone()

    m.disarm()
    m.reset()
    for p in layer.parameters():
        p.grad = None
    m.arm()
    checkpoint(block, x, use_reentrant=False).sum().backward()
    torch.testing.assert_close(m.sumsq, direct)


# ── through the trainer ──────────────────────────────────────────────

import json  # noqa: E402
import pathlib  # noqa: E402

from moestack.callbacks import Callback, ExpertGradStats  # noqa: E402
from moestack.trainer import Trainer  # noqa: E402

from conftest import tiny_config  # noqa: E402


class _Capture(Callback):
    def __init__(self):
        self.seen = []

    def on_step_end(self, trainer, state):
        self.seen.append(dict(state.metrics))


def _run(tmp_path, grad_every, steps=4, stats_every=0, **accum):
    cfg = tiny_config(tmp_path)
    cfg.train.n_steps = steps
    cfg.logging.grad_stats_every = grad_every
    cfg.logging.expert_stats_every = stats_every
    for k, v in accum.items():
        setattr(cfg.optim.expert_accum, k, v)
    cfg.__post_init__()
    cap = _Capture()
    t = Trainer(cfg, callbacks=[cap], verbose=False)
    t.fit()
    return t, cap


def _events(t, tmp_path):
    path = pathlib.Path(tmp_path) / f"{t.run_name}_grad_events.jsonl"
    return [json.loads(line) for line in path.open()] if path.exists() else []


def test_snr_reaches_the_dashboard_summarized_not_per_expert(tmp_path):
    _, cap = _run(tmp_path, grad_every=1, steps=2)
    m = cap.seen[-1]
    assert {"grad/snr_median", "grad/snr_p10", "grad/snr_max",
            "grad/expert_norm_median", "update/rel_median",
            "update/per_token_median"} <= set(m)
    # 16 x 64 would be 1,024 series; summarize caps it at eight per quantity
    assert sum(k.startswith("grad/snr") for k in m) == 8


def test_interval_and_event_displacement_are_separate_series(tmp_path):
    """`ExpertStats` measures net displacement over the reporting interval and
    `ExpertGradStats` measures one event's displacement. Both are ||dw||/||w||, and
    they shared the name `update/rel`: with ExpertStats registered last it won every
    reporting step except the first, where it returns early for want of a second
    snapshot, so the series silently alternated between two different quantities.
    Every other test here runs with `expert_stats_every=0`, which is why none of
    them saw it."""
    _, cap = _run(tmp_path, grad_every=1, steps=3, stats_every=1)
    m = cap.seen[-1]
    assert {"update/rel_median", "update/per_token_median"} <= set(m)   # event clock
    assert {"drift/rel_median", "drift/corr_share_vs_update"} <= set(m)  # step clock
    assert sum(k.startswith("update/rel") for k in m) == 8
    assert sum(k.startswith("drift/rel") for k in m) == 8


def test_grad_stats_respect_the_reporting_cadence(tmp_path):
    _, cap = _run(tmp_path, grad_every=3, steps=4)
    assert ["grad/snr_median" in m for m in cap.seen] == [True, False, False, True]


def test_nothing_is_measured_when_the_cadence_is_off(tmp_path):
    _, cap = _run(tmp_path, grad_every=0, steps=2)
    assert not any(k.startswith(("grad/", "update/")) for m in cap.seen for k in m)


def test_one_jsonl_row_per_expert_update_event(tmp_path):
    t, _ = _run(tmp_path, grad_every=1, steps=2)
    rows = _events(t, tmp_path)
    assert rows, "no event rows written"
    for r in rows:
        assert set(r) >= {"step", "layer", "expert", "tokens", "micro",
                          "snr", "grad_norm", "rel_update", "per_token"}
    # a baseline fires every expert once per step
    L, E = t.args.n_layers, t.args.n_experts
    assert len([r for r in rows if r["step"] == 0]) == L * E


def test_a_baseline_step_is_one_window_of_grad_accum_microbatches(tmp_path):
    """That is what anchors the SNR-versus-window-tokens curve at the
    un-accumulated end."""
    t, _ = _run(tmp_path, grad_every=1, steps=1)
    rows = _events(t, tmp_path)
    assert {r["micro"] for r in rows} == {t.cfg.data.grad_accum}


def test_a_baseline_event_carries_its_token_budget(tmp_path):
    """Without it `update/per_token` would be NaN for every baseline cell, and the
    metric exists precisely to compare baseline against accumulated windows."""
    t, _ = _run(tmp_path, grad_every=1, steps=1)
    rows = _events(t, tmp_path)
    assert any(r["tokens"] and r["tokens"] > 0 for r in rows)
    assert any(r["per_token"] == r["per_token"] for r in rows)      # not NaN


def test_displacement_is_non_zero_under_every_clock(tmp_path):
    """The bug this replaced: under clock=microbatch the experts move inside the
    micro-batch loop, so a step-level before/after bracketed an interval in which
    nothing happened and reported exactly 0."""
    for clock in (None, "microbatch", "step"):
        kw = {} if clock is None else dict(enabled=True, clock=clock,
                                           threshold=1, max_window=1)
        _, cap = _run(tmp_path / f"c{clock}", grad_every=1, steps=3, **kw)
        rel = cap.seen[-1]["update/rel_median"]
        assert rel > 0, f"clock={clock} reported zero displacement"


def test_accumulation_takes_over_the_clock(tmp_path):
    t, _ = _run(tmp_path, grad_every=1, steps=2, enabled=True,
                clock="microbatch", threshold=1, max_window=1)
    cb = next(c for c in t.callbacks.callbacks if isinstance(c, ExpertGradStats))
    assert cb._external


def test_the_grad_window_is_ticked_before_the_controller_closes_it(tmp_path):
    """ExpertGradStats is registered before ExpertAccumulation so its tick lands
    first. Registered the other way round, every window undercounts its own
    micro-batches by one."""
    t, _ = _run(tmp_path, grad_every=1, steps=1, enabled=True,
                clock="microbatch", threshold=1, max_window=1)
    order = [type(c).__name__ for c in t.callbacks.callbacks]
    assert order.index("ExpertGradStats") < order.index("ExpertAccumulation")
    assert all(r["micro"] >= 1 for r in _events(t, tmp_path))


# ── SNR must not depend on anything that rescales the gradient ───────

def _step0_snr(tmp_path, **accum):
    """Median SNR of the events at step 0, where every configuration starts from
    identical weights and sees identical data -- so the gradients are the same and
    any difference in SNR is a measurement artefact, not training."""
    import statistics
    t, _ = _run(tmp_path, grad_every=1, steps=1, **accum)
    vals = [r["snr"] for r in _events(t, tmp_path)
            if r["step"] == 0 and r["snr"] == r["snr"]]
    return statistics.median(sorted(vals))


def test_snr_is_the_same_under_every_clock_at_step_zero(tmp_path):
    """A baseline and a Q=1 step clock fire every expert every step on the same
    micro-batches, so they are the same measurement and must agree.

    They did not. Two separate things rescaled `p.grad` after the hooks had already
    recorded the raw per-micro-batch norms, scaling one side of the ratio only:
    `ExpertAccumulation._arm` divides the firing gradients by their token count,
    and `clip_grad_norm_` shrinks the baseline's expert gradients by a global
    factor (under accumulation the experts are accumulators and are excluded from
    clipping). SNR came out ~29x low in the first case and ~3.7x low in the second.
    """
    base = _step0_snr(tmp_path / "base")
    q1 = _step0_snr(tmp_path / "q1", enabled=True, clock="step",
                    threshold=1, max_window=99)
    assert q1 == pytest.approx(base, rel=1e-6)


def test_snr_does_not_depend_on_the_grad_scale_mode(tmp_path):
    """`grad_scale` changes the gradient handed to the optimizer, never the
    agreement among the micro-gradients that built it."""
    mean = _step0_snr(tmp_path / "mean", enabled=True, clock="step",
                      threshold=1, max_window=99, grad_scale="mean")
    total = _step0_snr(tmp_path / "sum", enabled=True, clock="step",
                       threshold=1, max_window=99, grad_scale="sum")
    assert mean == pytest.approx(total, rel=1e-9)


def test_snr_does_not_depend_on_the_clipping_threshold(tmp_path):
    """Clipping is one global scalar over the whole model, so it says nothing about
    any single expert's within-window agreement and must divide out."""
    def with_clip(c, d):
        cfg = tiny_config(d)
        cfg.train.n_steps, cfg.logging.grad_stats_every = 1, 1
        cfg.optim.grad_clip = c
        cfg.__post_init__()
        t = Trainer(cfg, verbose=False)
        t.fit()
        import statistics
        vals = [r["snr"] for r in _events(t, d) if r["snr"] == r["snr"]]
        return statistics.median(sorted(vals))

    assert with_clip(1.0, tmp_path / "a") == pytest.approx(
        with_clip(0.01, tmp_path / "b"), rel=1e-5)
