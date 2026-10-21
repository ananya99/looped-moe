"""Data parallelism on CPU (gloo, 2 processes).

1. DDP gradients on two half-batches == single-process gradients on the full batch.
2. A 2-rank Trainer keeps replicas bit-identical and reduces expert counts over
   the global batch.
3. Per-expert learning rates are derived identically on every rank.
4. A gradient used as a multi-batch accumulator survives DDP's bucket: never
   clearing it still accumulates correctly, and zeroing one expert's slice leaves
   the rest of the bucket alone.
"""

import os
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _setup(rank, world, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)


def _grad_worker(rank, world, port, out_dir):
    _setup(rank, world, port)
    from conftest import tiny_args
    from moestack.config import LossConfig, ParallelConfig
    from moestack.initialization import init_weights
    from moestack.losses import compute_loss
    from moestack.model import Transformer
    from moestack.parallel import init_distributed, parallelize

    d = init_distributed()
    torch.manual_seed(0)
    model = Transformer(tiny_args())
    init_weights(model)
    fwd = parallelize(model, ParallelConfig(), d)
    g = torch.Generator().manual_seed(123)
    x = torch.randint(0, 256, (4, 16), generator=g)
    y = torch.roll(x, -1, 1)
    xs, ys = x.chunk(world)[rank], y.chunk(world)[rank]
    cfg = LossConfig(aux_coef=0.0, z_coef=1e-3)        # per-token terms only (see test_training)
    compute_loss(fwd(xs), ys, cfg)[0].backward()
    if rank == 0:
        torch.save({n: p.grad for n, p in model.named_parameters()}, Path(out_dir) / "ddp.pt")
    dist.destroy_process_group()


def _trainer_worker(rank, world, port, out_dir):
    _setup(rank, world, port)
    from conftest import tiny_config
    from moestack.trainer import Trainer

    cfg = tiny_config(Path(out_dir), train=dict(n_steps=4))
    t = Trainer(cfg, verbose=False)
    seen = []

    class Rec:
        def __getattr__(self, name):
            return lambda *a: None

        def on_step_end(self, trainer, state):
            seen.append(int(state.expert_counts[0].sum()))

    t.callbacks.callbacks.append(Rec())
    t.fit()
    flat = torch.cat([p.detach().flatten() for p in t.model.parameters()])
    gathered = [torch.zeros_like(flat) for _ in range(world)]
    dist.all_gather(gathered, flat)
    if rank == 0:
        torch.save({"gathered": gathered, "seen": seen, "tokens_per_step": t.tokens_per_step},
                   Path(out_dir) / "trainer.pt")
    t.close()


@pytest.mark.slow
def test_ddp_gradients_equal_single_process(tmp_path):
    mp.spawn(_grad_worker, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True)
    ddp = torch.load(tmp_path / "ddp.pt")

    from conftest import tiny_args
    from moestack.config import LossConfig
    from moestack.initialization import init_weights
    from moestack.losses import compute_loss
    from moestack.model import Transformer
    torch.manual_seed(0)
    model = Transformer(tiny_args())
    init_weights(model)
    g = torch.Generator().manual_seed(123)
    x = torch.randint(0, 256, (4, 16), generator=g)
    compute_loss(model(x), torch.roll(x, -1, 1), LossConfig(aux_coef=0.0, z_coef=1e-3))[0].backward()
    for n, p in model.named_parameters():
        torch.testing.assert_close(ddp[n], p.grad, atol=1e-6, rtol=1e-4, msg=n)


@pytest.mark.slow
def test_two_rank_trainer_keeps_replicas_identical(tmp_path):
    mp.spawn(_trainer_worker, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True)
    r = torch.load(tmp_path / "trainer.pt")
    assert torch.equal(r["gathered"][0], r["gathered"][1])
    # counts are summed over both ranks' micro-batches: tokens_per_step already includes dp=2
    assert r["seen"] == [r["tokens_per_step"] * 2] * 4


def _per_expert_lr_worker(rank, world, port, out_dir):
    """Per-expert LR under DDP. The multipliers are not gradients, so DDP never
    synchronises them: if two ranks derived different rates they would take
    different steps from identical gradients and the replicas would drift apart
    with nothing raising. What makes them agree is that the trainer all-reduces
    the expert counts *before* any callback runs."""
    _setup(rank, world, port)
    from conftest import tiny_config
    from moestack.callbacks import PerExpertLR
    from moestack.trainer import Trainer

    cfg = tiny_config(Path(out_dir) / f"rank{rank}", train=dict(n_steps=3))
    cfg.optim.per_expert_lr.alpha = 0.5
    cfg.optim.per_expert_lr.freeze_dead = True   # deadness is decided from counts too
    cfg.__post_init__()
    t = Trainer(cfg, verbose=False)
    t.fit()

    cb = next(c for c in t.callbacks.callbacks if isinstance(c, PerExpertLR))
    scales = torch.stack([gs[0]["lr_scale"] for _, gs in sorted(t.optimizers.expert_lr_groups().items())])
    flat = torch.cat([p.detach().flatten() for p in t.model.parameters()])
    out = {}
    for name, tensor in (("scales", scales), ("ema", cb.ema), ("weights", flat)):
        buf = [torch.zeros_like(tensor) for _ in range(world)]
        dist.all_gather(buf, tensor.contiguous())
        out[name] = buf
    out["counts"] = t.args.n_experts
    if rank == 0:
        torch.save(out, Path(out_dir) / "per_expert_lr.pt")
    t.close()


@pytest.mark.slow
def test_per_expert_lr_is_identical_on_every_rank(tmp_path):
    mp.spawn(_per_expert_lr_worker, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True)
    r = torch.load(tmp_path / "per_expert_lr.pt")
    # Byte-identical, not close: these are derived from an all-reduced integer
    # count matrix by deterministic arithmetic, so any difference is a bug.
    assert torch.equal(r["scales"][0], r["scales"][1])
    assert torch.equal(r["ema"][0], r["ema"][1])
    # ... and the thing divergence would actually corrupt:
    assert torch.equal(r["weights"][0], r["weights"][1])
    # the rule must have been doing something, or this proves nothing
    assert r["scales"][0].shape[-1] == r["counts"]
    assert not torch.allclose(r["scales"][0], torch.ones_like(r["scales"][0]))


# ──────────────────────────────────────────────────────────────────────
# Expert-local accumulation keeps its window in `p.grad` and never clears
# it between micro-batches. Under DDP that gradient is a *view into the
# reducer's bucket* (parallelize passes gradient_as_bucket_view=True), so
# before anything is built on top of it: does a never-zeroed gradient
# still accumulate correctly, and does writing one expert's slice leave
# the others alone?
# ──────────────────────────────────────────────────────────────────────

def _probe_batches(n=3):
    g = torch.Generator().manual_seed(123)
    return [torch.randint(0, 256, (4, 16), generator=g) for _ in range(n)]


def _bucket_worker(rank, world, port, out_dir):
    _setup(rank, world, port)
    from conftest import tiny_args
    from moestack.config import LossConfig, ParallelConfig
    from moestack.initialization import init_weights
    from moestack.losses import compute_loss
    from moestack.model import Transformer
    from moestack.parallel import init_distributed, parallelize

    d = init_distributed()
    cfg = LossConfig(aux_coef=0.0, z_coef=1e-3)
    batches = _probe_batches()

    def run(zero_slice_after=None):
        torch.manual_seed(0)
        model = Transformer(tiny_args())
        init_weights(model)
        fwd = parallelize(model, ParallelConfig(), d)
        expert = next(iter(model.moe_layers())).experts.gate_up_proj
        running = []
        for i, x in enumerate(batches):
            xs, ys = x.chunk(world)[rank], torch.roll(x, -1, 1).chunk(world)[rank]
            # No no_sync: every micro-batch is reduced, which is what an expert
            # that may fire mid-batch needs. Nothing is ever zeroed.
            compute_loss(fwd(xs), ys, cfg)[0].backward()
            running.append(expert.grad.detach().clone())
            if i == zero_slice_after:
                expert.grad[0].zero_()      # expert 0 closes its window here
        return running, {n: p.grad.detach().clone() for n, p in model.named_parameters()}

    plain, plain_all = run()
    zeroed, _ = run(zero_slice_after=0)
    if rank == 0:
        torch.save({"plain": plain, "zeroed": zeroed, "all": plain_all},
                   Path(out_dir) / "bucket.pt")
    dist.destroy_process_group()


@pytest.mark.slow
def test_a_never_zeroed_gradient_still_accumulates_correctly_under_ddp(tmp_path):
    mp.spawn(_bucket_worker, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True)
    r = torch.load(tmp_path / "bucket.pt")

    from conftest import tiny_args
    from moestack.config import LossConfig
    from moestack.initialization import init_weights
    from moestack.losses import compute_loss
    from moestack.model import Transformer
    torch.manual_seed(0)
    model = Transformer(tiny_args())
    init_weights(model)
    cfg = LossConfig(aux_coef=0.0, z_coef=1e-3)
    for x in _probe_batches():
        compute_loss(model(x), torch.roll(x, -1, 1), cfg)[0].backward()

    # Re-reducing an already-reduced partial sum is a no-op under a mean
    # all-reduce, so syncing every micro-batch costs communication but changes no
    # arithmetic: three un-cleared micro-batches equal one single-process run.
    for n, p in model.named_parameters():
        torch.testing.assert_close(r["all"][n], p.grad, atol=1e-6, rtol=1e-4, msg=n)


@pytest.mark.slow
def test_closing_one_experts_window_leaves_every_other_experts_untouched(tmp_path):
    """`p.grad[e].zero_()` writes into the reducer's bucket. If that disturbed its
    neighbours -- or if the reducer overwrote rather than accumulated afterwards --
    every expert's window would be wrong in a way nothing would raise."""
    mp.spawn(_bucket_worker, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True)
    r = torch.load(tmp_path / "bucket.pt")
    plain_final, zeroed_final, after_first = r["plain"][-1], r["zeroed"][-1], r["plain"][0]

    assert torch.equal(zeroed_final[1:], plain_final[1:])
    # Expert 0 kept exactly what arrived after its reset, and nothing from before.
    torch.testing.assert_close(zeroed_final[0], plain_final[0] - after_first[0],
                               atol=1e-6, rtol=1e-4)
    assert not torch.allclose(zeroed_final[0], plain_final[0])   # the reset did something
