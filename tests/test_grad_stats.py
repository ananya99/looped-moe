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


def test_snr_reaches_the_dashboard_summarized_not_per_expert(tmp_path):
    _, cap = _run(tmp_path, grad_every=1, steps=2)
    m = cap.seen[0]
    assert {"grad/snr_median", "grad/snr_p10", "grad/snr_max",
            "grad/expert_norm_median"} <= set(m)
    # 16 x 64 would be 1,024 series; summarize caps it at seven per quantity
    assert sum(k.startswith("grad/snr") for k in m) == 8


def test_grad_stats_respect_the_cadence(tmp_path):
    _, cap = _run(tmp_path, grad_every=3, steps=4)
    assert ["grad/snr_median" in m for m in cap.seen] == [True, False, False, True]


def test_nothing_is_measured_when_the_cadence_is_off(tmp_path):
    _, cap = _run(tmp_path, grad_every=0, steps=2)
    assert not any(k.startswith("grad/") for m in cap.seen for k in m)


def test_the_matrices_ride_along_in_the_expert_stats_jsonl(tmp_path):
    t, _ = _run(tmp_path, grad_every=1, steps=2, stats_every=1)
    path = tmp_path / f"{t.run_name}_expert_stats.jsonl"
    records = [json.loads(line) for line in path.open()]
    L, E = t.args.n_layers, t.args.n_experts
    for r in records:
        for key in ("grad_snr", "grad_grad_norm", "grad_micro"):
            assert len(r[key]) == L and len(r[key][0]) == E, key


def test_a_real_step_measures_a_window_of_grad_accum_microbatches(tmp_path):
    """A baseline step *is* a window -- that is what anchors the SNR-vs-window
    curve at the un-accumulated end."""
    t, _ = _run(tmp_path, grad_every=1, steps=1, stats_every=1)
    cb = next(c for c in t.callbacks.callbacks if isinstance(c, ExpertGradStats))
    assert (cb.last["micro"] == t.cfg.data.grad_accum).all()


def test_accumulation_takes_over_the_clock(tmp_path):
    """With expert-local accumulation the controller owns window boundaries, so the
    step boundary must not blanket-reset them."""
    t, _ = _run(tmp_path, grad_every=1, steps=2, enabled=True, threshold=1, max_window=1)
    cb = next(c for c in t.callbacks.callbacks if isinstance(c, ExpertGradStats))
    assert cb._external_clock
