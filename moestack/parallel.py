"""Where the model meets the hardware. The model code never imports this module.

    init_distributed()            -> Dist (rank, world, device)
    ParallelDims.build_mesh()     -> named DeviceMesh views
    parallelize(model, ...)       -> AC -> compile -> DDP, in that fixed order

Only data parallelism is implemented. The mesh already has the two views an MoE
needs -- a *dense* view for attention/embeddings/router and an *expert* view
(edp x ep) for expert weights -- so FSDP2 and expert parallelism are additions to
`parallelize` (plus an all-to-all `Dispatcher`), not a restructuring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from .config import ParallelConfig


@dataclass
class Dist:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def init_distributed(device: str | None = None) -> Dist:
    """torchrun sets RANK/WORLD_SIZE; otherwise run single-process."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            dev = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(dev)
        else:
            dev = torch.device("cpu")
        return Dist(rank, local_rank, world, dev)
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Dist(0, 0, 1, dev)


@dataclass
class ParallelDims:
    dp: int
    ep: int
    tp: int
    world_size: int

    @classmethod
    def from_config(cls, cfg: ParallelConfig, world_size: int) -> "ParallelDims":
        if cfg.ep != 1 or cfg.tp != 1:
            raise NotImplementedError(
                "Only data parallelism is implemented. EP plugs in as an all-to-all "
                "Dispatcher plus expert-dim sharding in parallelize(); TP as a plan in "
                "parallelize(). See moestack/README.md.")
        dp = cfg.dp if cfg.dp != -1 else world_size // (cfg.ep * cfg.tp)
        dims = cls(dp=dp, ep=cfg.ep, tp=cfg.tp, world_size=world_size)
        if dp * cfg.ep * cfg.tp != world_size:
            raise ValueError(f"dp*ep*tp = {dp}*{cfg.ep}*{cfg.tp} != world_size {world_size}")
        return dims

    def build_mesh(self, device_type: str):
        """Returns {"dense": mesh, "expert": mesh} or None when single-process.
        dense  = ("dp",)          -- what attention/router/embeddings reduce over
        expert = ("edp", "ep")    -- expert weights: replicated over edp, sharded over ep
        With ep = 1 both reduce over the same ranks."""
        if self.world_size == 1:
            return None
        from torch.distributed.device_mesh import init_device_mesh
        dense = init_device_mesh(device_type, (self.dp,), mesh_dim_names=("dp",))
        expert = init_device_mesh(device_type, (self.world_size // self.ep, self.ep),
                                  mesh_dim_names=("edp", "ep"))
        return {"dense": dense, "expert": expert}


def apply_activation_checkpointing(model: nn.Module) -> None:
    """Non-reentrant checkpointing of every transformer block. Safe for routing
    statistics because they are read from the block's *return value*.

    Applied in place (the block's `forward` is wrapped) rather than with a wrapper
    module, so parameter names -- and therefore checkpoints -- do not change."""
    from torch.utils.checkpoint import checkpoint
    for block in model.layers:
        inner = block.forward

        def forward(*args, _inner=inner, **kwargs):
            return checkpoint(_inner, *args, use_reentrant=False, **kwargs)

        block.forward = forward


def apply_compile(model: nn.Module) -> None:
    """Per-block compile (torchtitan convention): small graphs, reused across blocks.
    `Module.compile()` is in place, so parameter names do not change either."""
    for block in model.layers:
        block.compile()


def parallelize(model: nn.Module, cfg: ParallelConfig, d: Dist) -> nn.Module:
    """Returns the module to call forward on. `model` itself stays the unwrapped
    module that owns the parameters (use it for optimizers and checkpoints)."""
    if cfg.activation_checkpointing:
        apply_activation_checkpointing(model)
    if cfg.compile:
        apply_compile(model)
    if d.enabled:
        from torch.nn.parallel import DistributedDataParallel as DDP
        return DDP(model, device_ids=[d.local_rank] if d.device.type == "cuda" else None,
                   broadcast_buffers=False, gradient_as_bucket_view=True)
    return model


def all_reduce_(t: torch.Tensor, op=None) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=op or dist.ReduceOp.SUM)
    return t
