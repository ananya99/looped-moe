"""End-to-end behaviour of the training step, the loop, checkpointing and the
activation-checkpointing / compile transforms (CPU, synthetic data)."""

import copy

import pytest
import torch

from moestack.config import LossConfig, OptimConfig, ParallelConfig
from moestack.initialization import init_weights
from moestack.losses import compute_loss
from moestack.model import Transformer
from moestack.optim import build_optimizers
from moestack.parallel import Dist, parallelize
from moestack.trainer import Trainer

from conftest import tiny_args, tiny_config

CPU = Dist(0, 0, 1, torch.device("cpu"))


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    m = Transformer(tiny_args(**kw))
    init_weights(m)
    return m


def test_overfits_a_single_batch():
    m = _model()
    opts = build_optimizers(m, OptimConfig(name="muon", lr=0.02))
    x = torch.randint(0, 256, (4, 32))
    y = torch.roll(x, -1, dims=1)
    first = None
    for _ in range(150):
        opts.zero_grad()
        loss, terms = compute_loss(m(x), y, LossConfig())
        loss.backward()
        opts.step()
        first = first if first is not None else terms[0].value.item()
    last = terms[0].value.item()
    assert first > 5.0 and last < 0.5, (first, last)


def test_grad_accumulation_equals_full_batch():
    """With equal micro-batch sizes, mean-per-micro / accum == mean over the full
    batch for per-token terms (lm, z). The balance term is NOT linear in the batch
    (it multiplies two batch means), so it is switched off here -- and gradient
    accumulation does change it, as it did in the original trainer."""
    cfg = LossConfig(aux_coef=0.0, z_coef=1e-3)
    m1 = _model()
    m2 = copy.deepcopy(m1)
    x = torch.randint(0, 256, (4, 16))
    y = torch.roll(x, -1, dims=1)
    compute_loss(m1(x), y, cfg)[0].backward()
    for xs, ys in zip(x.chunk(2), y.chunk(2)):
        (compute_loss(m2(xs), ys, cfg)[0] / 2).backward()
    for (n, a), b in zip(m1.named_parameters(), m2.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4, msg=n)


def test_activation_checkpointing_is_numerically_transparent():
    m1 = _model()
    m2 = copy.deepcopy(m1)
    fwd2 = parallelize(m2, ParallelConfig(activation_checkpointing=True), CPU)
    assert m2.state_dict().keys() == m1.state_dict().keys()      # names unchanged
    x = torch.randint(0, 256, (2, 16))
    o1, o2 = m1(x), fwd2(x)
    compute_loss(o1, x, LossConfig())[0].backward()
    compute_loss(o2, x, LossConfig())[0].backward()
    # routing counts come from the original forward only (not doubled by recompute)
    for a, b in zip(o1.router_outs, o2.router_outs):
        assert torch.equal(a.counts, b.counts)
    for (n, a), b in zip(m1.named_parameters(), m2.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5, msg=n)


def test_trainer_runs_and_loss_decreases(tmp_path):
    cfg = tiny_config(tmp_path, train=dict(n_steps=30), optim=dict(lr=0.02, warmup_steps=5))
    t = Trainer(cfg, verbose=False)
    losses = []

    class Rec:
        def __getattr__(self, name):
            return lambda *a: None

        def on_step_end(self, trainer, state):
            losses.append(state.loss_terms["lm"])
            assert state.expert_counts.shape == (2, 8)
            # every token of the global batch is assigned top_k experts in every layer
            assert int(state.expert_counts[0].sum()) == trainer.tokens_per_step * 2
            assert "load/gini" in state.metrics

    t.callbacks.callbacks.insert(1, Rec())
    res = t.fit()
    assert losses[-1] < losses[0] - 0.5
    assert (tmp_path / f"{t.run_name}_step30.pt").exists()
    assert (tmp_path / f"{t.run_name}_results.json").exists()
    assert res["tokens_seen"] == 30 * t.tokens_per_step


def test_resume_reproduces_uninterrupted_run(tmp_path):
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    straight = Trainer(tiny_config(a_dir, train=dict(n_steps=6)), verbose=False)
    straight.fit()

    first = Trainer(tiny_config(b_dir, train=dict(n_steps=6, save_every=3)), verbose=False)
    first.cfg.train.n_steps = 3                  # stop early; schedule still spans 6
    while first.step < 3:
        first.train_step()
        first.step += 1
    first.save("step3")

    cfg = tiny_config(b_dir, train=dict(n_steps=6))
    cfg.train.resume = str(b_dir / f"{first.run_name}_step3.pt")
    resumed = Trainer(cfg, verbose=False)
    assert resumed.step == 3 and resumed.train_iter.consumed == 3 * cfg.data.grad_accum
    resumed.fit()

    sa, sb = straight.model.state_dict(), resumed.model.state_dict()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    oa, ob = straight.optimizers.state_dict(), resumed.optimizers.state_dict()
    for name in oa:
        for pid in oa[name]["state"]:
            for key, v in oa[name]["state"][pid].items():
                assert torch.equal(torch.as_tensor(v), torch.as_tensor(ob[name]["state"][pid][key]))


def test_unimplemented_parallelism_fails_loudly(tmp_path):
    cfg = tiny_config(tmp_path, parallel=dict(ep=2))
    with pytest.raises(NotImplementedError):
        Trainer(cfg, verbose=False)


@pytest.mark.slow
def test_compile_matches_eager():
    m1 = _model()
    m2 = copy.deepcopy(m1)
    fwd2 = parallelize(m2, ParallelConfig(compile=True), CPU)
    x = torch.randint(0, 256, (2, 16))
    torch.testing.assert_close(m1(x).logits, fwd2(x).logits, atol=1e-4, rtol=1e-4)
