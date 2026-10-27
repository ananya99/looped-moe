"""Compass alignment factors (moestack/compass.py), measured not applied.

Pins the arithmetic of arXiv:2610.04140 eqs. 2-6 on CPU, and the one property that
makes this safe to switch on mid-grid: the probe must not change the update.
"""
import math

import pytest
import torch

from moestack.compass import (alignment, expert_scale, family_factor, radius,
                              row_alignment)


def test_a_perfectly_aligned_family_has_every_factor_at_one():
    torch.manual_seed(0)
    o = torch.randn(8, 16, 16)
    out = expert_scale(o, o, lam=1.0)
    for k, v in out.items():
        torch.testing.assert_close(v, torch.ones(8), atol=1e-5, rtol=1e-5,
                                   msg=lambda m, k=k: f"{k}: {m}")


def test_alignment_is_a_clamped_cosine_per_expert():
    o = torch.zeros(2, 2, 2)
    o[0, 0, 0] = 1.0
    o[1, 0, 0] = 1.0
    g = o.clone()
    g[1] *= -1                                  # anti-aligned -> clamped to 0
    torch.testing.assert_close(alignment(o, g), torch.tensor([1.0, 0.0]))


def test_the_family_factor_averages_to_one_over_the_family():
    """Both normalisations together redistribute step length without changing the
    total -- that is what makes it a reweighting rather than a global LR change."""
    torch.manual_seed(1)
    a = torch.rand(16)
    c = family_factor(a)
    assert float(c.mean()) == pytest.approx(1.0, rel=1e-5)


def test_the_family_factor_can_exceed_one_because_the_cap_pulls_the_mean_down():
    """v_e = min(1, a_e/abar) caps the well-aligned experts, so the second mean is
    below one and dividing by it lifts them back above it. This is why the paper's
    expert scale reaches 1.05-1.10 while the radius never exceeds 1."""
    a = torch.full((8,), 0.05)
    a[0] = 1.0
    c = family_factor(a)
    assert float(c[0]) > 1.0 and float(c[1]) < 1.0


def test_dead_experts_are_excluded_from_the_family_mean():
    """Their gradient is exactly zero, so their 'alignment' is not a measurement and
    must not drag the mean every other expert is normalised against."""
    a = torch.tensor([0.8, 0.8, 0.0, 0.0])
    alive = torch.tensor([True, True, False, False])
    assert float(family_factor(a, alive)[0]) == pytest.approx(1.0, rel=1e-5)
    assert float(family_factor(a)[0]) > 1.0          # without the mask it is skewed


def test_the_radius_never_exceeds_one():
    torch.manual_seed(2)
    o, g = torch.randn(8, 16, 16), torch.randn(8, 16, 16)
    a = radius(o, row_alignment(o, g), lam=1.0)
    assert bool(((a >= 0) & (a <= 1)).all())


def test_the_radius_relaxes_to_one_as_the_learning_rate_decays():
    """h_er = 1 + lam(q_er - 1), so at lam = 0 every row factor is exactly 1 and the
    radius stops shortening anything."""
    torch.manual_seed(3)
    o, g = torch.randn(4, 8, 8), torch.randn(4, 8, 8)
    b = row_alignment(o, g)
    torch.testing.assert_close(radius(o, b, lam=0.0), torch.ones(4), atol=1e-6, rtol=1e-6)
    assert float(radius(o, b, lam=1.0).mean()) < 1.0


def test_the_radius_preserves_the_direction_identity():
    """Paper eq. 6: ||alpha_e O_e||_F == ||diag(h_e) O_e||_F. The scalar takes only
    the *length* of the row-weighted matrix, so Muon's direction is untouched."""
    torch.manual_seed(4)
    o, g, lam = torch.randn(3, 6, 6), torch.randn(3, 6, 6), 0.7
    b = row_alignment(o, g)
    q = (b / b.mean(dim=-1, keepdim=True)).clamp(max=1.0)
    h = 1.0 + lam * (q - 1.0)
    lhs = radius(o, b, lam) * o.flatten(1).norm(dim=1)
    rhs = (h.unsqueeze(-1) * o).flatten(1).norm(dim=1)
    torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)


# ── the probe must not change training ───────────────────────────────

def test_the_probe_does_not_change_the_update(tmp_path):
    """The whole point of measuring rather than applying: two runs that differ only
    in whether CompassStats is on must produce identical weights."""
    import sys
    sys.path.insert(0, "tests")
    from conftest import tiny_config
    from moestack.trainer import Trainer

    def weights(compass_every, d):
        cfg = tiny_config(d)
        cfg.train.n_steps = 3
        cfg.logging.compass_stats_every = compass_every
        cfg.__post_init__()
        t = Trainer(cfg, verbose=False)
        t.fit()
        return [p.detach().clone() for p in t.model.parameters()]

    off = weights(0, tmp_path / "off")
    on = weights(1, tmp_path / "on")
    for a, b in zip(off, on):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
