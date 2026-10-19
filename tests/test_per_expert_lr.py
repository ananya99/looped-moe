"""Per-expert LR rules: the share -> multiplier math (moestack/expert_lr.py)."""

import pytest
import torch

from moestack.expert_lr import ema_update, lr_scales_from_counts, scale_summary, token_shares

ALPHAS = [0.0, 0.5, 1.0, -0.5, 2.0]


def _counts(*rows):
    return torch.tensor(rows, dtype=torch.long)


# ── shares ───────────────────────────────────────────────────────────

def test_shares_are_one_when_balanced():
    torch.testing.assert_close(token_shares(_counts([8, 8, 8, 8])), torch.ones(1, 4))


def test_shares_of_a_layer_with_no_tokens_are_ones_not_nan():
    s = token_shares(_counts([0, 0, 0, 0]))
    assert torch.isfinite(s).all() and (s == 1).all()


# ── the baseline guarantee ───────────────────────────────────────────

def test_alpha_zero_without_freeze_is_exactly_uniform():
    c = _counts([100, 1, 0, 7], [3, 3, 3, 3])
    s = lr_scales_from_counts(c, alpha=0.0, freeze_dead=False)
    assert (s == 1.0).all()


@pytest.mark.parametrize("alpha", ALPHAS)
def test_balanced_counts_give_uniform_multipliers_at_every_alpha(alpha):
    s = lr_scales_from_counts(_counts([5, 5, 5, 5]), alpha=alpha)
    torch.testing.assert_close(s, torch.ones(1, 4))


# ── the rule itself ──────────────────────────────────────────────────

def test_alpha_one_is_token_proportional_up_to_the_normalisation():
    c = _counts([1, 2, 3, 4])
    s = lr_scales_from_counts(c, alpha=1.0, normalize=True)
    ratios = s[0] / c[0].float()
    torch.testing.assert_close(ratios, ratios[0].expand(4))


def test_positive_alpha_favours_hot_experts_and_negative_alpha_favours_cold_ones():
    c = _counts([1, 2, 4, 8])
    hot = lr_scales_from_counts(c, alpha=0.5)[0]
    cold = lr_scales_from_counts(c, alpha=-0.5)[0]
    assert torch.all(hot.diff() > 0)        # increasing with token count
    assert torch.all(cold.diff() < 0)       # decreasing with token count


def test_normalisation_averages_to_one_over_alive_experts_only():
    c = _counts([1, 2, 0, 8, 0])
    s = lr_scales_from_counts(c, alpha=1.0, normalize=True)[0]
    alive = c[0] > 0
    torch.testing.assert_close(s[alive].mean(), torch.tensor(1.0))
    assert (s[~alive] == 1.0).all()         # dead experts sit at the uniform rate


def test_without_normalisation_alpha_also_moves_the_average_rate():
    # alpha=1 is the one exponent that preserves the mean by itself (mean(s) == 1),
    # so the confound normalisation removes only shows up at other alphas.
    c = _counts([1, 2, 4, 8])
    s = lr_scales_from_counts(c, alpha=0.5, normalize=False)[0]
    assert abs(float(s.mean()) - 1.0) > 1e-2
    n = lr_scales_from_counts(c, alpha=0.5, normalize=True)[0]
    torch.testing.assert_close(n.mean(), torch.tensor(1.0))


def test_rows_are_independent_across_layers():
    c = _counts([1, 2, 4, 8], [4, 4, 4, 4])
    s = lr_scales_from_counts(c, alpha=1.0)
    assert not torch.allclose(s[0], torch.ones(4))
    torch.testing.assert_close(s[1], torch.ones(4))


# ── dead experts ─────────────────────────────────────────────────────

@pytest.mark.parametrize("alpha", ALPHAS)
def test_dead_experts_never_produce_inf_or_nan(alpha):
    s = lr_scales_from_counts(_counts([0, 0, 5, 9]), alpha=alpha)
    assert torch.isfinite(s).all()


def test_freeze_dead_zeroes_exactly_the_zero_token_experts():
    c = _counts([0, 3, 0, 9])
    s = lr_scales_from_counts(c, alpha=0.0, freeze_dead=True)[0]
    assert (s == torch.tensor([0.0, 1.0, 0.0, 1.0])).all()


def test_deadness_is_keyed_on_raw_counts_not_on_the_ema():
    """An expert with EMA history that received nothing *this* step is dead."""
    raw = _counts([0, 4, 4, 4])
    ema = torch.tensor([[3.0, 4.0, 4.0, 4.0]])       # expert 0 still carries mass
    frozen = lr_scales_from_counts(ema, alpha=0.5, freeze_dead=True, dead_counts=raw)[0]
    assert frozen[0] == 0.0
    # Without dead_counts the EMA hides it and the freeze silently never fires.
    assert lr_scales_from_counts(ema, alpha=0.5, freeze_dead=True)[0][0] > 0.0


def test_mismatched_dead_counts_shape_is_an_error():
    with pytest.raises(ValueError):
        lr_scales_from_counts(_counts([1, 2, 3, 4]), alpha=1.0, dead_counts=torch.ones(1, 3))


# ── clamping ─────────────────────────────────────────────────────────

def test_clamp_is_applied_and_is_asymmetric():
    # one expert takes nearly everything: its raw multiplier is far above the ceiling
    c = _counts([1, 1, 1, 100_000])
    s = lr_scales_from_counts(c, alpha=2.0, normalize=False, clamp_min=1e-3, clamp_max=10.0)[0]
    assert float(s.max()) == pytest.approx(10.0)
    assert float(s.min()) >= 1e-3


# ── EMA ──────────────────────────────────────────────────────────────

def test_ema_is_seeded_by_the_first_observation():
    c = _counts([4, 8])
    torch.testing.assert_close(ema_update(None, c, 0.9), c.float())


def test_ema_converges_to_a_constant_stream():
    c = _counts([4.0, 8.0])
    e = None
    for _ in range(200):
        e = ema_update(e, c, 0.9)
    torch.testing.assert_close(e, c.float(), rtol=1e-4, atol=1e-4)


# ── summary metrics ──────────────────────────────────────────────────

def test_summary_reports_frozen_fraction_and_ignores_dead_in_the_alive_mean():
    c = _counts([0, 0, 4, 4])
    s = lr_scales_from_counts(c, alpha=0.0, freeze_dead=True)
    m = scale_summary(s, c)
    assert m["lr_scale/frozen_fraction"] == pytest.approx(0.5)
    assert m["lr_scale/mean_alive"] == pytest.approx(1.0)
    assert m["lr_scale/min"] == 0.0


# ── parity with the original scalar implementation ───────────────────

@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("freeze", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_matches_expert_stats_compute_lr_scales_layer_by_layer(alpha, freeze, normalize):
    """The vectorised rule must be the original rule, row by row -- that is what
    makes a moestack run comparable with an imbalance_train.py run."""
    original = pytest.importorskip("expert_stats")       # repo root, torch only
    torch.manual_seed(0)
    counts = torch.randint(0, 40, (5, 16))
    counts[0, :3] = 0                                     # dead experts
    counts[1] = 0                                         # a layer with no tokens
    counts[2, 7] = 5000                                   # one expert past the ceiling
    ema = counts.float() * 0.7 + 1.0                      # EMA-like, never exactly zero

    mine = lr_scales_from_counts(ema, alpha=alpha, freeze_dead=freeze,
                                 normalize=normalize, dead_counts=counts)
    for li in range(counts.size(0)):
        ref = original.compute_lr_scales(ema[li], alpha=alpha, freeze_dead=freeze,
                                         normalize=normalize, dead_counts=counts[li])
        torch.testing.assert_close(mine[li], ref, msg=f"layer {li}")


# ──────────────────────────────────────────────────────────────────────
# The multiplier inside Muon (moestack/optim/muon.py)
# ──────────────────────────────────────────────────────────────────────

from moestack.optim import Muon  # noqa: E402

MUON_KW = dict(lr=0.02, weight_decay=0.1, momentum=0.95, nesterov=True)
E, OUT, IN = 4, 24, 16


def _grads(n_steps, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(E, OUT, IN, generator=g) for _ in range(n_steps)]


def _run(param, scales, grads, **kw):
    opt = Muon([param], **{**MUON_KW, **kw})
    if scales is not None:
        opt.param_groups[0]["lr_scale"] = scales
    for g in grads:
        param.grad = g.clone()
        opt.step()
    return param


def test_all_ones_scale_is_bit_identical_to_no_scale():
    """The baseline must reduce exactly, not approximately: at alpha=0 every
    multiplier is 1.0, and a run with per-expert LR on has to match one with it off."""
    grads = _grads(5)
    w0 = torch.randn(E, OUT, IN)
    plain = _run(torch.nn.Parameter(w0.clone()), None, grads)
    scaled = _run(torch.nn.Parameter(w0.clone()), torch.ones(E), grads)
    assert torch.equal(plain.data, scaled.data)


def test_per_expert_scale_equals_running_each_expert_at_its_own_lr():
    scales = torch.tensor([0.25, 1.0, 2.0, 0.5])
    grads = _grads(4, seed=1)
    w0 = torch.randn(E, OUT, IN)

    got = _run(torch.nn.Parameter(w0.clone()), scales, grads)
    for e, s in enumerate(scales.tolist()):
        # One expert on its own, at lr * s_e: adjusted_lr is linear in lr, so this
        # is the exact per-matrix reference.
        ref = _run(torch.nn.Parameter(w0[e: e + 1].clone()), None,
                   [g[e: e + 1] for g in grads], lr=MUON_KW["lr"] * s)
        torch.testing.assert_close(got.data[e], ref.data[0], msg=f"expert {e}")


def test_zero_scale_freezes_an_expert_completely_including_weight_decay():
    """freeze_dead is only a real control if the decoupled weight decay stops too:
    otherwise a dead expert keeps shrinking every step."""
    scales = torch.tensor([0.0, 1.0, 1.0, 1.0])
    w0 = torch.randn(E, OUT, IN)
    out = _run(torch.nn.Parameter(w0.clone()), scales, _grads(6, seed=2))
    assert torch.equal(out.data[0], w0[0])
    assert not torch.allclose(out.data[1], w0[1])


def test_a_frozen_expert_keeps_its_momentum_and_moves_once_unfrozen():
    """Matches the original, where freezing meant pg['lr'] = 0: the buffer still
    accumulates, so unfreezing resumes from the accumulated direction."""
    grads = _grads(3, seed=3)
    w0 = torch.randn(E, OUT, IN)
    p = torch.nn.Parameter(w0.clone())
    opt = Muon([p], **MUON_KW)
    opt.param_groups[0]["lr_scale"] = torch.tensor([0.0, 1.0, 1.0, 1.0])
    for g in grads:
        p.grad = g.clone()
        opt.step()
    assert torch.equal(p.data[0], w0[0])
    buf = opt.state[p]["momentum_buffer"]
    assert buf[0].abs().sum() > 0                 # the buffer did accumulate
    opt.param_groups[0]["lr_scale"] = torch.ones(E)
    p.grad = torch.zeros_like(p)                  # no new signal: pure stale momentum
    opt.step()
    assert not torch.equal(p.data[0], w0[0])


def test_scale_survives_an_optimizer_state_dict_round_trip():
    """checkpoint.py saves Optimizers.state_dict(); a resumed run must not silently
    fall back to uniform rates between the restore and the first callback call."""
    scales = torch.tensor([0.5, 1.5, 1.0, 0.0])
    p = torch.nn.Parameter(torch.randn(E, OUT, IN))
    opt = Muon([p], **MUON_KW)
    opt.param_groups[0]["lr_scale"] = scales
    sd = opt.state_dict()

    p2 = torch.nn.Parameter(torch.randn(E, OUT, IN))
    opt2 = Muon([p2], **MUON_KW)
    opt2.load_state_dict(sd)
    torch.testing.assert_close(opt2.param_groups[0]["lr_scale"], scales)


def test_lr_scale_on_a_2d_parameter_is_an_error():
    p = torch.nn.Parameter(torch.randn(OUT, IN))
    opt = Muon([p], **MUON_KW)
    opt.param_groups[0]["lr_scale"] = torch.ones(OUT)
    p.grad = torch.randn(OUT, IN)
    with pytest.raises(ValueError, match="3D"):
        opt.step()


def test_lr_scale_of_the_wrong_length_is_an_error():
    p = torch.nn.Parameter(torch.randn(E, OUT, IN))
    opt = Muon([p], **MUON_KW)
    opt.param_groups[0]["lr_scale"] = torch.ones(E + 1)
    p.grad = torch.randn(E, OUT, IN)
    with pytest.raises(ValueError, match="1D tensor of length"):
        opt.step()


# ──────────────────────────────────────────────────────────────────────
# Addressing: one param group per expert tensor (moestack/optim/builder.py)
# ──────────────────────────────────────────────────────────────────────

from moestack.config import OptimConfig  # noqa: E402
from moestack.model import Transformer  # noqa: E402
from moestack.optim import build_optimizers  # noqa: E402

from conftest import tiny_args  # noqa: E402


def _model():
    torch.manual_seed(0)
    return Transformer(tiny_args())


def _expert_group_count(opts):
    return sum(1 for g in opts.param_groups if "expert" in g["roles"] and g.get("layer") is not None)


def test_default_rules_are_unchanged_without_split():
    opts = build_optimizers(_model(), OptimConfig())
    assert opts.expert_lr_groups() == {}
    assert _expert_group_count(opts) == 0


def test_split_gives_one_group_per_expert_tensor_tagged_with_its_layer():
    m = _model()
    opts = build_optimizers(m, OptimConfig(), split_experts=True)
    groups = opts.expert_lr_groups()
    assert sorted(groups) == list(range(m.args.n_layers))
    # gate_up and down of the same layer, both addressable under that layer
    for layer, gs in groups.items():
        assert len(gs) == 2
        assert {g["name"].rsplit(".", 1)[-1] for g in gs} == {"gate_up_proj", "down_proj"}
        assert all(len(g["params"]) == 1 and g["params"][0].dim() == 3 for g in gs)


def test_splitting_moves_no_parameter_between_optimizers():
    m = _model()
    plain = build_optimizers(m, OptimConfig())
    split = build_optimizers(m, OptimConfig(), split_experts=True)
    for name in ("muon", "adamw"):
        a = {id(p) for g in plain.optimizers[name].param_groups for p in g["params"]}
        b = {id(p) for g in split.optimizers[name].param_groups for p in g["params"]}
        assert a == b, name


def test_splitting_alone_changes_nothing_numerically():
    """Group layout is bookkeeping: with no multipliers set, a split run and an
    unsplit run must be the same run."""
    grads = {}
    outs = []
    for split in (False, True):
        m = _model()
        opts = build_optimizers(m, OptimConfig(), split_experts=split)
        for step in range(3):
            g = torch.Generator().manual_seed(step)
            for i, p in enumerate(m.parameters()):
                key = (step, i)
                if key not in grads:
                    grads[key] = torch.randn(p.shape, generator=g)
                p.grad = grads[key].clone()
            opts.set_lr(0.02)
            opts.step()
        outs.append([p.detach().clone() for p in m.parameters()])
    for a, b in zip(*outs):
        assert torch.equal(a, b)


def test_set_expert_lr_scales_puts_each_row_on_its_own_layer():
    m = _model()
    opts = build_optimizers(m, OptimConfig(), split_experts=True)
    L, E = m.args.n_layers, m.args.n_experts
    scales = torch.arange(L * E, dtype=torch.float32).reshape(L, E)
    opts.set_expert_lr_scales(scales)
    for layer, gs in opts.expert_lr_groups().items():
        for g in gs:
            torch.testing.assert_close(g["lr_scale"], scales[layer])
    # nothing else acquired a multiplier
    others = [g for g in opts.param_groups if g.get("layer") is None]
    assert all(g.get("lr_scale") is None for g in others)


def test_per_expert_lr_under_adamw_raises_instead_of_silently_doing_nothing():
    """Under AdamW the experts stay single 3D tensors with one scalar lr, so the
    rule cannot be applied; the original raised here too."""
    opts = build_optimizers(_model(), OptimConfig(name="adamw"), split_experts=True)
    with pytest.raises(RuntimeError, match="optim.name=muon"):
        opts.set_expert_lr_scales(torch.ones(2, 8))


def test_scales_with_too_few_layers_raise():
    m = _model()
    opts = build_optimizers(m, OptimConfig(), split_experts=True)
    with pytest.raises(ValueError, match="n_layers"):
        opts.set_expert_lr_scales(torch.ones(1, m.args.n_experts))


# ──────────────────────────────────────────────────────────────────────
# Config, validation and run identity (moestack/config.py)
# ──────────────────────────────────────────────────────────────────────

from moestack.config import Config, PerExpertLRConfig  # noqa: E402


def _cfg(**per_expert):
    c = Config()
    for k, v in per_expert.items():
        setattr(c.optim.per_expert_lr, k, v)
    c.__post_init__()          # re-run validation after the overrides
    return c


def test_defaults_are_off_and_do_not_touch_existing_run_names():
    """Adding an option must not rename runs that never set it, or every existing
    checkpoint becomes unreachable."""
    c = Config()
    assert not c.optim.per_expert_lr.enabled
    assert c.significant_overrides() == {}
    assert c.fingerprint() == "base"


@pytest.mark.parametrize("kw,expected", [
    (dict(alpha=0.0), False),
    (dict(alpha=0.5), True),
    (dict(alpha=-0.5), True),
    (dict(freeze_dead=True), True),
    (dict(ema=0.0), False),           # smoothing alone changes nothing
    (dict(normalize=False), False),
])
def test_enabled_tracks_only_the_knobs_that_change_the_update(kw, expected):
    assert PerExpertLRConfig(**kw).enabled is expected


@pytest.mark.parametrize("kw,fragment", [
    (dict(alpha=0.5), "-a0.5-"),
    (dict(alpha=1.0), "-a1-"),
    (dict(alpha=-0.5), "-aneg0.5-"),      # a bare minus would read as a separator
    (dict(freeze_dead=True), "-freeze-"),
    (dict(alpha=0.5, freeze_dead=True), "-a0.5-freeze-"),
])
def test_condition_name_spells_out_the_cell(kw, fragment):
    assert fragment in _cfg(**kw).condition_name()


def test_every_matrix_cell_gets_its_own_run_name():
    cells = [dict(), dict(alpha=0.5), dict(alpha=1.0), dict(alpha=-0.5),
             dict(freeze_dead=True), dict(alpha=0.5, ema=0.0),
             dict(alpha=0.5, normalize=False), dict(alpha=0.5, clamp_max=2.0)]
    names = {_cfg(**c).run_name() for c in cells}
    assert len(names) == len(cells)


def test_nested_overrides_are_reported_field_by_field():
    c = _cfg(alpha=0.5, ema=0.0)
    assert c.significant_overrides() == {"optim.per_expert_lr.alpha": 0.5,
                                         "optim.per_expert_lr.ema": 0.0}


def test_per_expert_lr_under_adamw_fails_at_startup():
    c = Config()
    c.optim.name, c.optim.per_expert_lr.alpha = "adamw", 1.0
    with pytest.raises(ValueError, match="optim.name=muon"):
        c.__post_init__()


@pytest.mark.parametrize("kw", [dict(ema=1.0), dict(ema=-0.1),
                                dict(clamp_min=5.0, clamp_max=1.0), dict(clamp_min=-1.0)])
def test_invalid_knobs_fail_at_startup(kw):
    with pytest.raises(ValueError):
        _cfg(**kw)


# ──────────────────────────────────────────────────────────────────────
# The callback end to end (moestack/callbacks.py + trainer wiring)
# ──────────────────────────────────────────────────────────────────────

from moestack.callbacks import PerExpertLR, StepState  # noqa: E402
from moestack.trainer import Trainer  # noqa: E402

from conftest import tiny_config  # noqa: E402


def _train(tmp_path, steps=3, **per_expert):
    cfg = tiny_config(tmp_path)
    cfg.train.n_steps = steps
    for k, v in per_expert.items():
        setattr(cfg.optim.per_expert_lr, k, v)
    cfg.__post_init__()
    t = Trainer(cfg, verbose=False)
    t.fit()
    return t


def test_trainer_only_builds_the_callback_when_a_knob_is_turned(tmp_path):
    off = _train(tmp_path / "off", steps=1)
    assert not any(isinstance(cb, PerExpertLR) for cb in off.callbacks.callbacks)
    assert off.optimizers.expert_lr_groups() == {}

    on = _train(tmp_path / "on", steps=1, alpha=0.5)
    assert any(isinstance(cb, PerExpertLR) for cb in on.callbacks.callbacks)
    assert on.optimizers.expert_lr_groups()


def test_an_enabled_run_with_no_dead_experts_matches_the_baseline(tmp_path):
    """freeze_dead with every expert alive leaves all multipliers at 1.0, so the
    machinery must be inert: same weights as a run without it, bit for bit."""
    base = _train(tmp_path / "base")
    frozen = _train(tmp_path / "frozen", freeze_dead=True)
    # Guard the premise: if this config ever starts producing dead experts, the
    # weights legitimately diverge and the failure should say why.
    for gs in frozen.optimizers.expert_lr_groups().values():
        assert (gs[0]["lr_scale"] == 1.0).all(), "premise broken: an expert went dead"
    for a, b in zip(base.model.parameters(), frozen.model.parameters()):
        assert torch.equal(a, b)


def test_scales_reach_the_optimizer_and_the_metrics(tmp_path):
    t = _train(tmp_path, alpha=1.0)
    L, E = t.args.n_layers, t.args.n_experts
    for layer, gs in t.optimizers.expert_lr_groups().items():
        for g in gs:
            s = g["lr_scale"]
            assert s.shape == (E,) and torch.isfinite(s).all() and (s > 0).all()
    cb = next(c for c in t.callbacks.callbacks if isinstance(c, PerExpertLR))
    assert cb.ema.shape == (L, E)


def test_metrics_are_published_for_the_loggers(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg.train.n_steps = 1
    cfg.optim.per_expert_lr.alpha = 0.5
    cfg.__post_init__()
    t = Trainer(cfg, verbose=False)
    state = t.train_step()
    assert {"lr_scale/min", "lr_scale/max", "lr_scale/mean_alive",
            "lr_scale/frozen_fraction"} <= set(state.metrics)


def test_ema_is_checkpointed_and_restored_warm(tmp_path):
    t = _train(tmp_path, alpha=0.5, steps=3)
    cb = next(c for c in t.callbacks.callbacks if isinstance(c, PerExpertLR))
    saved = cb.ema.clone()
    assert (saved > 0).any()

    fresh = PerExpertLR(t.cfg.optim.per_expert_lr)
    assert fresh.ema is None
    fresh.load_state_dict(t.callbacks.state_dict()["PerExpertLR"])
    torch.testing.assert_close(fresh.ema, saved)


def test_ema_zero_runs_the_rule_on_raw_counts(tmp_path):
    t = _train(tmp_path, alpha=0.5, ema=0.0, steps=2)
    cb = next(c for c in t.callbacks.callbacks if isinstance(c, PerExpertLR))
    assert cb.ema is None          # never built, so nothing to checkpoint
    assert t.callbacks.state_dict()["PerExpertLR"] == {"ema": None}


def test_callback_refuses_optimizers_without_addressable_experts(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg.train.n_steps = 1
    t = Trainer(cfg, verbose=False)          # built WITHOUT split_experts
    with pytest.raises(RuntimeError, match="split_experts=True"):
        PerExpertLR(cfg.optim.per_expert_lr).on_train_start(t)


def test_summary_reports_rates_relative_to_and_in_units_of_the_base_lr():
    c = _counts([1, 2, 4, 8])
    s = lr_scales_from_counts(c, alpha=1.0)
    m = scale_summary(s, c, base_lr=0.02)
    # lr_scale/* are ratios to the base rate; expert_lr/* are the same in LR units
    assert m["expert_lr/base"] == pytest.approx(0.02)
    assert m["expert_lr/max"] == pytest.approx(m["lr_scale/max"] * 0.02)
    assert m["expert_lr/mean_alive"] == pytest.approx(0.02)     # normalisation holds it there
    assert m["lr_scale/spread_alive"] == pytest.approx(m["lr_scale/max"] / m["lr_scale/min"])


def test_spread_is_one_when_the_rule_does_nothing():
    c = _counts([1, 2, 4, 8])
    m = scale_summary(lr_scales_from_counts(c, alpha=0.0), c)
    assert m["lr_scale/spread_alive"] == pytest.approx(1.0)
    assert "expert_lr/base" not in m          # absolute units only when asked for


def test_frozen_experts_do_not_poison_the_alive_spread():
    c = _counts([0, 2, 4, 8])
    m = scale_summary(lr_scales_from_counts(c, alpha=0.5, freeze_dead=True), c, base_lr=0.02)
    assert m["lr_scale/min"] == 0.0 and m["expert_lr/min"] == 0.0
    assert m["lr_scale/min_alive"] > 0 and torch.isfinite(torch.tensor(m["lr_scale/spread_alive"]))


def test_absolute_expert_lr_follows_the_schedule(tmp_path):
    """expert_lr/* must track train/lr, so the two can be read on one dashboard."""
    cfg = tiny_config(tmp_path)
    cfg.train.n_steps, cfg.optim.per_expert_lr.alpha = 1, 1.0
    cfg.__post_init__()
    t = Trainer(cfg, verbose=False)
    state = t.train_step()
    assert state.metrics["expert_lr/base"] == pytest.approx(state.lr)
    assert state.metrics["expert_lr/max"] >= state.metrics["expert_lr/mean_alive"] > 0


def test_resume_keeps_a_warm_ema_and_reproduces_the_uninterrupted_run(tmp_path):
    """Cluster jobs are timeout-capped and restarted, so a 2000-step run resumes
    several times. A cold EMA at each restart would swing the multipliers for ~10
    steps every time -- a discontinuity in the intervention, with nothing raising.

    This exercises the full path the unit test does not: checkpoint.save ->
    torch.load(map_location="cpu") -> CallbackList.load_state_dict, which hands the
    callback a CPU tensor while the counts live on the training device.
    """
    def cfg_for(d, **train):
        c = tiny_config(d, train=dict(n_steps=6, **train))
        c.optim.per_expert_lr.alpha = 0.5
        c.__post_init__()
        return c

    def ema_of(trainer):
        return next(c for c in trainer.callbacks.callbacks if isinstance(c, PerExpertLR)).ema

    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    straight = Trainer(cfg_for(a_dir), verbose=False)
    straight.fit()

    first = Trainer(cfg_for(b_dir, save_every=3), verbose=False)
    first.cfg.train.n_steps = 3                  # stop early; the LR schedule still spans 6
    while first.step < 3:
        first.train_step()
        first.step += 1
    first.save("step3")
    warm = ema_of(first).clone()

    cfg = cfg_for(b_dir)
    cfg.train.resume = str(b_dir / f"{first.run_name}_step3.pt")
    resumed = Trainer(cfg, verbose=False)
    # Restored before the first step of the resumed run, not re-seeded from it.
    torch.testing.assert_close(ema_of(resumed), warm)
    resumed.fit()

    torch.testing.assert_close(ema_of(resumed), ema_of(straight))
    sa, sb = straight.model.state_dict(), resumed.model.state_dict()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    for layer, gs in straight.optimizers.expert_lr_groups().items():
        other = resumed.optimizers.expert_lr_groups()[layer]
        torch.testing.assert_close(gs[0]["lr_scale"], other[0]["lr_scale"])


def test_resuming_into_a_different_per_expert_lr_setting_says_why_it_cannot(tmp_path):
    """Enabling the rule splits the expert tensors into one param group per layer,
    so the optimizer state of a per-expert-LR run cannot load into a baseline run.
    Run names differ too, so this needs an explicit resume path to hit -- but when
    it is hit the message should name the cause, not just the symptom."""
    cfg = tiny_config(tmp_path)
    cfg.train.n_steps, cfg.optim.per_expert_lr.alpha = 2, 0.5
    cfg.__post_init__()
    t = Trainer(cfg, verbose=False)
    t.fit()
    t.save("step2")

    plain = tiny_config(tmp_path)
    plain.train.n_steps = 4
    plain.train.resume = str(tmp_path / f"{t.run_name}_step2.pt")
    with pytest.raises(ValueError, match="optim.per_expert_lr"):
        Trainer(plain, verbose=False)


def test_the_ema_entry_is_ignored_when_no_callback_claims_it(tmp_path):
    """CallbackList keys state by class name, so a checkpoint carrying a
    PerExpertLR entry loads fine into a run without that callback -- the entry is
    simply unclaimed. Silent, therefore pinned."""
    from moestack.callbacks import CallbackList, MoELoadStats
    cbs = CallbackList([MoELoadStats()])
    cbs.load_state_dict({"PerExpertLR": {"ema": torch.ones(2, 8)}, "MoELoadStats": {}})
    assert cbs.state_dict() == {"MoELoadStats": {}}
