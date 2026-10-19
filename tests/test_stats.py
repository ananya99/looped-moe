"""Per-expert measurement: routing inequality and weight drift (moestack/stats.py),
plus the ExpertStats callback that writes the per-expert JSONL."""

import json
import math

import pytest
import torch

from moestack.callbacks import Callback, ExpertStats
from moestack.stats import (delta_norms, drift_metrics, gini, load_metrics,
                            normalized_entropy, row_norms, spearman)
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

def test_load_metrics_reports_layer_series_and_their_means():
    c = torch.tensor([[0, 0, 4, 12], [4, 4, 4, 4]])
    m = load_metrics(c)
    assert m["load_layer/dead_fraction/00"] == pytest.approx(0.5)
    assert m["load_layer/dead_fraction/01"] == pytest.approx(0.0)
    assert m["load/dead_expert_fraction"] == pytest.approx(0.25)
    assert m["load_layer/gini/01"] == pytest.approx(0.0)
    assert m["load/gini_max_layer"] == pytest.approx(m["load_layer/gini/00"])


def test_per_layer_series_can_be_switched_off():
    m = load_metrics(torch.tensor([[1, 2, 3, 4]]), per_layer=False)
    assert not any("load_layer/" in k for k in m)
    assert "load/gini" in m


def test_a_layer_with_no_tokens_is_skipped_rather_than_counted_as_balanced():
    m = load_metrics(torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]]))
    assert "load_layer/gini/00" not in m
    assert m["load/gini"] == pytest.approx(m["load_layer/gini/01"])


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


def test_jsonl_holds_one_record_per_measured_step_shaped_layers_by_experts(tmp_path):
    t, _, records = _run(tmp_path, every=2, steps=4, alpha=0.5)
    assert len(records) == 2                               # steps 0 and 2
    L, E = t.args.n_layers, t.args.n_experts
    for r in records:
        for key in ("counts", "drift", "drift_gate_up", "drift_down",
                    "weight_norm_gate_up", "weight_norm_down", "lr_scale"):
            assert len(r[key]) == L and len(r[key][0]) == E, key
        assert sum(sum(row) for row in r["counts"]) > 0


def test_the_jsonl_is_the_wide_data_wandb_cannot_hold(tmp_path):
    """Dashboard gets correlations; the file gets the joint distribution they are
    computed from, per expert."""
    _, cap, records = _run(tmp_path, every=1, steps=2)
    assert {"drift/corr_share_vs_update", "drift/mean"} <= set(cap.seen[0])
    assert not any(k.startswith("drift/") and "layer" in k for k in cap.seen[0])
    counts = torch.tensor(records[0]["counts"])
    drift = torch.tensor(records[0]["drift"])
    assert counts.shape == drift.shape                     # recomputable offline


def test_no_file_and_no_drift_metrics_when_the_cadence_is_off(tmp_path):
    _, cap, records = _run(tmp_path, every=0, steps=2)
    assert records == []
    assert not any(k.startswith("drift/") for m in cap.seen for k in m)


def test_drift_is_only_reported_on_cadence_steps(tmp_path):
    _, cap, _ = _run(tmp_path, every=3, steps=4)
    has_drift = ["drift/mean" in m for m in cap.seen]
    assert has_drift == [True, False, False, True]


def test_a_dropped_step_is_not_recorded_as_zero_drift(tmp_path):
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
