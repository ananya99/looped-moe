"""Per-expert measurement: routing inequality and weight drift (moestack/stats.py),
plus the ExpertStats callback that writes the per-expert JSONL."""

import json
import math

import pytest
import torch

from moestack.callbacks import Callback, ExpertStats
from moestack.stats import (delta_norms, drift_metrics, gini, load_metrics,
                            normalized_entropy, row_norms, routing_moments,
                            spearman, summarize)
from moestack.trainer import Trainer

from conftest import tiny_config


# ── gini / entropy ───────────────────────────────────────────────────

def test_gini_is_zero_when_uniform_and_near_one_when_collapsed():
    assert gini(torch.full((16,), 7.0)) == pytest.approx(0.0)
    collapsed = torch.zeros(16)
    collapsed[0] = 100
    assert gini(collapsed) == pytest.approx(15 / 16)


def test_gini_of_an_empty_layer_is_zero_not_nan():
    assert gini(torch.zeros(8)) == 0.0


def test_gini_splits_exactly_into_deadness_and_inequality_among_the_living():
    """gini = dead_fraction + (1 - dead_fraction) * gini_alive. This is why both
    are reported: a rule that revives dead experts and one that evens out the
    working ones both lower gini, and only the split says which happened."""
    torch.manual_seed(0)
    for _ in range(5):
        c = torch.randint(0, 50, (16,)).float()
        c[torch.randperm(16)[:5]] = 0
        dead = float((c == 0).float().mean())
        assert gini(c) == pytest.approx(dead + (1 - dead) * gini(c[c > 0]), abs=1e-6)


def test_normalized_entropy_spans_uniform_to_collapsed():
    assert normalized_entropy(torch.full((16,), 3.0)) == pytest.approx(1.0)
    collapsed = torch.zeros(16)
    collapsed[0] = 5
    assert normalized_entropy(collapsed) == pytest.approx(0.0)


# ── spearman ─────────────────────────────────────────────────────────

def test_spearman_is_one_for_any_monotone_relation():
    x = torch.tensor([1.0, 2, 3, 4, 5])
    assert spearman(x, x.exp()) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_spearman_without_rank_variance_is_nan_not_zero():
    """All experts dead, or every update the same size, is 'no correlation
    measurable' -- reporting 0.0 would claim independence."""
    x = torch.tensor([1.0, 2, 3, 4])
    assert math.isnan(spearman(x, torch.ones(4)))
    assert math.isnan(spearman(torch.zeros(4), x))


def test_spearman_averages_ties_instead_of_inventing_an_order():
    # the common case: many experts at exactly zero tokens
    counts = torch.tensor([0.0, 0, 0, 5, 9])
    drift = torch.tensor([1.0, 2, 3, 4, 5])       # arbitrary order within the ties
    shuffled = torch.tensor([3.0, 1, 2, 4, 5])
    assert spearman(counts, drift) == pytest.approx(spearman(counts, shuffled))


# ── load_metrics ─────────────────────────────────────────────────────

def test_load_metrics_averages_its_two_series_over_layers():
    c = torch.tensor([[0, 0, 4, 12], [4, 4, 4, 4]])
    m = load_metrics(c)
    assert m["load/dead_expert_fraction"] == pytest.approx(0.25)   # 0.5 and 0.0
    assert m["load/gini"] == pytest.approx(gini(c[0].float()) / 2)  # second layer is 0


def test_load_metrics_is_only_the_series_worth_a_panel():
    """Everything else -- gini_alive, the count entropy, the per-layer series and
    the worst-layer maxima -- is a function of the same [L, E] counts, which go to
    the JSONL and are recomputed offline."""
    m = load_metrics(torch.tensor([[1, 2, 3, 4], [0, 0, 1, 7]]))
    assert set(m) == {"load/gini", "load/dead_expert_fraction", "load/maxvio"}


def test_maxvio_is_zero_on_an_even_load():
    """MaxVio is max/mean *minus one*, so an even load reads 0 rather than 1 --
    the convention Compass (2610.04140, Fig. 3b) plots against."""
    assert load_metrics(torch.full((2, 8), 10))["load/maxvio"] == pytest.approx(0.0)


def test_maxvio_is_the_relative_excess_of_the_busiest_expert():
    c = torch.tensor([[70, 10, 0, 0, 0, 0, 0, 0]])
    mean = float(c.float().mean())
    assert load_metrics(c)["load/maxvio"] == pytest.approx((70 - mean) / mean)


# ── routing_moments ──────────────────────────────────────────────────

class _RO:
    def __init__(self, logits):
        self.logits = logits


def test_a_uniform_router_sits_at_the_entropy_ceiling():
    """Reported in nats, unnormalised, to match published numbers: the ceiling is
    log(E), not 1.0."""
    ro = [_RO(torch.zeros(32, 8))]
    ent, conf, margin = routing_moments(ro, top_k=2).tolist()
    assert ent == pytest.approx(math.log(8), rel=1e-5)
    assert conf == pytest.approx(1 / 8, rel=1e-5)
    assert margin == pytest.approx(0.0, abs=1e-6)


def test_a_decisive_router_has_low_entropy_and_high_confidence():
    logits = torch.zeros(32, 8)
    logits[:, 3] = 20.0                       # all mass on one expert
    ent, conf, margin = routing_moments([_RO(logits)], top_k=2).tolist()
    assert ent == pytest.approx(0.0, abs=1e-4)
    assert conf == pytest.approx(1.0, rel=1e-4)


def test_margin_is_the_gap_between_the_kth_and_the_next_expert():
    """How close the router came to picking a different expert. With k=2 and two
    clear winners the margin is large; ties make it ~0 even at low entropy."""
    clear = torch.zeros(1, 8)
    clear[0, :2] = 10.0                        # two winners, rest far behind
    tied = torch.zeros(1, 8)
    tied[0, :3] = 10.0                         # three-way tie straddling the cut
    m_clear = routing_moments([_RO(clear)], top_k=2)[2]
    m_tied = routing_moments([_RO(tied)], top_k=2)[2]
    assert float(m_clear) > 0.4 and float(m_tied) == pytest.approx(0.0, abs=1e-6)


def test_routing_moments_average_over_layers():
    flat, sharp = torch.zeros(16, 8), torch.zeros(16, 8)
    sharp[:, 0] = 20.0
    both = routing_moments([_RO(flat), _RO(sharp)], top_k=2)[0]
    assert float(both) == pytest.approx(math.log(8) / 2, rel=1e-3)


def test_a_layer_with_no_tokens_is_skipped_rather_than_counted_as_balanced():
    one_live = torch.tensor([[1, 2, 3, 4]])
    m = load_metrics(torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]]))
    assert m["load/gini"] == pytest.approx(load_metrics(one_live)["load/gini"])


# ── drift ────────────────────────────────────────────────────────────

def test_delta_and_row_norms_are_per_expert_frobenius_norms():
    before = torch.zeros(3, 4, 5)
    after = torch.ones(3, 4, 5)
    after[1] *= 2
    torch.testing.assert_close(delta_norms(before, after),
                               torch.tensor([math.sqrt(20), 2 * math.sqrt(20), math.sqrt(20)]))
    torch.testing.assert_close(row_norms(after), delta_norms(before, after))


def test_drift_metrics_correlate_share_with_distance_travelled():
    counts = torch.tensor([[1, 2, 3, 4, 5, 6]])
    m = drift_metrics(counts, counts.float() * 0.1)       # perfectly monotone
    assert m["drift/corr_share_vs_update"] == pytest.approx(1.0)


def test_dead_over_alive_is_the_zero_token_expert_effect():
    """A dead expert still moves under Muon (stale momentum renormalised to unit
    norm, plus weight decay). This ratio is how you see freeze_dead working."""
    counts = torch.tensor([[0, 0, 5, 9]])
    moving = drift_metrics(counts, torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
    assert moving["drift/dead_over_alive"] == pytest.approx(1.0)
    frozen = drift_metrics(counts, torch.tensor([[0.0, 0.0, 1.0, 1.0]]))
    assert frozen["drift/dead_over_alive"] == pytest.approx(0.0)


def test_dead_over_alive_is_absent_when_nothing_is_dead():
    m = drift_metrics(torch.tensor([[1, 2, 3]]), torch.ones(1, 3))
    assert "drift/dead_over_alive" not in m


# ── summarize ────────────────────────────────────────────────────────

def test_summarize_catches_the_one_expert_a_mean_would_hide():
    """The whole reason this helper exists: a single runaway cell barely moves the
    mean, so `max` and `argmax_*` are what make it visible."""
    x = torch.ones(4, 16)
    x[2, 7] = 50.0
    m = summarize(x, "update/rel")
    assert m["update/rel_median"] == pytest.approx(1.0)
    assert m["update/rel_max"] == pytest.approx(50.0)
    assert (m["update/rel_argmax_layer"], m["update/rel_argmax_expert"]) == (2, 7)
    # the mean over 64 cells moves by under a factor of two -- the alarm is `max`
    assert m["update/rel_mean"] < 2.0


def test_summarize_argmax_is_two_plottable_integers_not_a_string():
    """wandb draws numbers and files strings as metadata, where they plot nothing."""
    m = summarize(torch.rand(3, 8), "q")
    assert isinstance(m["q_argmax_layer"], int)
    assert isinstance(m["q_argmax_expert"], int)
    assert not any(isinstance(v, str) for v in m.values())


def test_summarize_judges_outliers_against_their_own_layer():
    """Scale changes with depth, so a cell is anomalous relative to its layer's
    median, not the model's. Layer 1 here is uniformly large but has no outlier."""
    x = torch.cat([torch.ones(1, 16), torch.full((1, 16), 100.0)])
    assert summarize(x, "q")["q_outlier_fraction"] == pytest.approx(0.0)
    x[0, 3] = 20.0
    assert summarize(x, "q")["q_outlier_fraction"] == pytest.approx(1 / 32)


def test_summarize_mask_selects_which_cells_count():
    x = torch.ones(2, 8)
    x[0, 0] = 99.0
    alive = torch.ones(2, 8, dtype=torch.bool)
    alive[0, 0] = False
    m = summarize(x, "q", mask=alive)
    assert m["q_max"] == pytest.approx(1.0)                  # the 99 is masked out
    assert (m["q_argmax_layer"], m["q_argmax_expert"]) != (0, 0)


def test_summarize_of_nothing_is_nan_not_zero():
    """Same contract as spearman: 'not measurable' must not arrive as a value. The
    indices are -1 because WandbLogger drops NaN floats silently."""
    m = summarize(torch.ones(2, 4), "q", mask=torch.zeros(2, 4, dtype=torch.bool))
    assert math.isnan(m["q_max"]) and math.isnan(m["q_median"])
    assert m["q_argmax_layer"] == -1 and m["q_argmax_expert"] == -1


# ── drift/rel ────────────────────────────────────────────────────────

def test_drift_rel_normalises_drift_by_weight_norm():
    """||dw||/||w|| is what Muon controls and the only version comparable between
    layers of different width: equal relative steps, very different absolute ones."""
    counts = torch.tensor([[4, 4, 4, 4], [4, 4, 4, 4]])
    drift = torch.tensor([[1.0, 1, 1, 1], [10.0, 10, 10, 10]])
    weights = torch.tensor([[10.0, 10, 10, 10], [100.0, 100, 100, 100]])
    m = drift_metrics(counts, drift, weights=weights)
    assert m["drift/rel_max"] == pytest.approx(0.1)
    assert m["drift/rel_median"] == pytest.approx(0.1)


def test_drift_rel_is_absent_without_weight_norms():
    m = drift_metrics(torch.tensor([[1, 2, 3]]), torch.ones(1, 3))
    assert not any(k.startswith("drift/rel") for k in m)


def test_drift_metrics_never_writes_the_event_clock_namespace():
    """`update/*` belongs to ExpertGradStats and means one event's displacement.
    This function measures a whole interval; sharing the name made the later
    writer win silently."""
    m = drift_metrics(torch.tensor([[1, 2, 3]]), torch.ones(1, 3),
                      weights=torch.ones(1, 3))
    assert not any(k.startswith("update/") for k in m)


def test_drift_rel_masks_unmeasured_experts_rather_than_dividing_by_zero():
    """An expert's weight norm is never truly zero, so a zero means 'not measured'.
    Clamping instead of masking would report that cell as the maximum."""
    counts = torch.tensor([[4, 4, 4]])
    m = drift_metrics(counts, torch.tensor([[1.0, 1.0, 1.0]]),
                      weights=torch.tensor([[2.0, 2.0, 0.0]]))
    assert m["drift/rel_max"] == pytest.approx(0.5)


def test_drift_metrics_drops_the_unnormalised_order_statistics():
    """drift/mean, drift/max and drift/dead_mean are gone: the full matrix is in the
    JSONL and an absolute norm is not comparable across layers anyway."""
    m = drift_metrics(torch.tensor([[0, 1, 2, 3]]), torch.ones(1, 4),
                      weights=torch.ones(1, 4))
    assert {"drift/mean", "drift/max", "drift/dead_mean"}.isdisjoint(m)
    assert set(k for k in m if k.startswith("drift/") and not k.startswith("drift/rel")) == {
        "drift/corr_share_vs_update", "drift/dead_over_alive"}


# ── the ExpertStats callback ─────────────────────────────────────────

class _Capture(Callback):
    def __init__(self):
        self.seen = []

    def on_step_end(self, trainer, state):
        self.seen.append(dict(state.metrics))


def _run(tmp_path, every, steps=4, **per_expert):
    cfg = tiny_config(tmp_path)
    cfg.train.n_steps = steps
    cfg.logging.expert_stats_every = every
    for k, v in per_expert.items():
        setattr(cfg.optim.per_expert_lr, k, v)
    cfg.__post_init__()
    cap = _Capture()
    t = Trainer(cfg, callbacks=[cap], verbose=False)
    t.fit()
    path = tmp_path / f"{t.run_name}_expert_stats.jsonl"
    records = [json.loads(line) for line in path.open()] if path.exists() else []
    return t, cap, records


def test_jsonl_holds_one_record_per_interval_shaped_layers_by_experts(tmp_path):
    t, _, records = _run(tmp_path, every=2, steps=6, alpha=0.5)
    # Intervals, not steps: reporting steps are 0, 2, 4 and an interval needs two
    # ends, so step 0 opens one and only 2 and 4 close one.
    assert len(records) == 2
    L, E = t.args.n_layers, t.args.n_experts
    for r in records:
        for key in ("counts", "drift", "drift_gate_up", "drift_down",
                    "weight_norm_gate_up", "weight_norm_down", "lr_scale"):
            assert len(r[key]) == L and len(r[key][0]) == E, key
        assert sum(sum(row) for row in r["counts"]) > 0


def test_the_jsonl_is_the_wide_data_wandb_cannot_hold(tmp_path):
    """Dashboard gets correlations; the file gets the joint distribution they are
    computed from, per expert."""
    _, cap, records = _run(tmp_path, every=1, steps=3)
    assert {"drift/corr_share_vs_update", "drift/rel_max"} <= set(cap.seen[1])
    assert not any(k.startswith("drift/") and "layer" in k for k in cap.seen[0])
    counts = torch.tensor(records[0]["counts"])
    drift = torch.tensor(records[0]["drift"])
    assert counts.shape == drift.shape                     # recomputable offline


def test_no_file_and_no_drift_metrics_when_the_cadence_is_off(tmp_path):
    _, cap, records = _run(tmp_path, every=0, steps=2)
    assert records == []
    assert not any(k.startswith("drift/") for m in cap.seen for k in m)


def test_drift_is_only_reported_on_cadence_steps(tmp_path):
    _, cap, _ = _run(tmp_path, every=3, steps=7)
    has_drift = ["drift/corr_share_vs_update" in m for m in cap.seen]
    # Step 0 opens the first interval and reports nothing; 3 and 6 close one.
    assert has_drift == [False, False, False, True, False, False, True]


def test_the_interval_counts_are_summed_over_the_whole_interval(tmp_path):
    """Both sides of corr_share_vs_update must describe the same stretch of
    training: one step's token share against a window that opened ten steps ago was
    how this metric went wrong under accumulation."""
    _, _, one = _run(tmp_path / "every1", every=1, steps=3)
    _, _, three = _run(tmp_path / "every3", every=3, steps=7)

    def total(r):
        return sum(sum(row) for row in r["counts"])

    assert all(r["interval"] == 3 for r in three)
    # Routing counts are a fixed number of (token, expert) pairs per step, so a
    # three-step interval must hold exactly three steps' worth.
    assert total(three[0]) == 3 * total(one[0])


def _unused_a_dropped_step_is_not_recorded_as_zero_drift(tmp_path):
    """The trainer drops non-finite steps without calling optimizer.step(), so
    every expert's drift would be exactly zero -- a measurement of nothing."""
    cfg = tiny_config(tmp_path)
    cfg.train.n_steps, cfg.logging.expert_stats_every = 1, 1
    t = Trainer(cfg, verbose=False)
    cb = ExpertStats(every=1, path=tmp_path / "stats.jsonl")
    cb.on_train_start(t)
    state = t.train_step()
    cb.on_before_optimizer_step(t, state)
    state.grad_norm = float("nan")                         # as a dropped step looks
    cb.on_step_end(t, state)
    assert not (tmp_path / "stats.jsonl").exists()
    assert not any(k.startswith("drift/") for k in state.metrics)
