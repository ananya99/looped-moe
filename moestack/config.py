"""Typed configuration tree.

Every option lives in one of these dataclasses. Hydra composes `conf/baseline.yaml`
on top of them (see `train.py`), so a typo in an override fails at startup instead of
being silently ignored.

Architecture presets reproduce the HF configs used by `imbalance_train.py` /
`moe_train.py` (`get_{olmoe,deepseek,qwen3moe}_config`) exactly, so a run here and a
run there describe the same network.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ModelArgs:
    vocab_size: int = 100277
    dim: int = 512
    n_layers: int = 16
    n_heads: int = 4
    n_kv_heads: int = 4
    head_dim: Optional[int] = None          # None = dim // n_heads
    n_experts: int = 64
    top_k: int = 6
    moe_inter_dim: int = 256                # per-expert SwiGLU hidden width
    norm_topk_prob: bool = False            # renormalise the k selected gate weights
    qk_norm: Literal["full", "head", "none"] = "full"   # OLMoE: full width; Qwen3: per head
    clip_qkv: Optional[float] = 8.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    max_seq_len: int = 4096
    experts_impl: Literal["grouped_mm", "loop"] = "grouped_mm"

    @property
    def hd(self) -> int:
        return self.head_dim or self.dim // self.n_heads


def _olmoe_like(scale: str, top_k: int) -> ModelArgs:
    tiny, small = scale == "tiny", scale == "small"
    heads = 4 if tiny else (8 if small else 16)
    return ModelArgs(
        dim=512 if tiny else (1024 if small else 2048),
        moe_inter_dim=256 if tiny else (512 if small else 1024),
        n_layers=16, n_heads=heads, n_kv_heads=heads,
        n_experts=64, top_k=top_k, norm_topk_prob=False,
        qk_norm="full", clip_qkv=8.0, rope_theta=10000.0, norm_eps=1e-5,
    )


def _qwen3moe(scale: str) -> ModelArgs:
    tiny, small = scale == "tiny", scale == "small"
    return ModelArgs(
        dim=384 if tiny else (768 if small else 1536),
        moe_inter_dim=192 if tiny else (384 if small else 768),
        n_layers=28,
        n_heads=6 if tiny else (12 if small else 24),
        n_kv_heads=1 if tiny else (2 if small else 4),
        n_experts=60, top_k=4, norm_topk_prob=True,
        qk_norm="head", clip_qkv=None, rope_theta=10000.0, norm_eps=1e-6,
    )


ARCHS = {
    # name: (description, builder)
    "olmoe": ("OLMoE-1B-7B-style, top-8", lambda s: _olmoe_like(s, top_k=8)),
    # Same misnomer as the original code: OLMoE layout with top-6. No MLA, no
    # fine-grained experts, no shared expert.
    "deepseek": ("OLMoE layout, top-6 ('deepseek' in the old code)", lambda s: _olmoe_like(s, top_k=6)),
    "qwen3moe": ("Qwen3-MoE-style", _qwen3moe),
}


def model_args_for(arch: str, scale: str, **overrides) -> ModelArgs:
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {list(ARCHS)}, got {arch!r}")
    if scale not in ("tiny", "small", "regular"):
        raise ValueError(f"scale must be tiny | small | regular, got {scale!r}")
    args = ARCHS[arch][1](scale)
    for k, v in overrides.items():
        if v is not None:
            setattr(args, k, v)
    return args


# ──────────────────────────────────────────────────────────────────────
# Everything else
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    arch: str = "deepseek"
    scale: str = "tiny"
    experts_impl: str = "grouped_mm"
    # Escape hatch for tests / quick experiments: override any ModelArgs field.
    overrides: Dict[str, Any] = field(default_factory=dict)

    def build_args(self) -> ModelArgs:
        return model_args_for(self.arch, self.scale, experts_impl=self.experts_impl,
                              **self.overrides)


@dataclass
class LossConfig:
    aux_coef: float = 0.01       # load-balancing (Switch/Fedus), mean over layers
    z_coef: float = 1e-4         # router z-loss (Zoph et al.), mean over layers
    # Memory knob, not an objective change: rows per chunk of the LM cross-entropy
    # (0 = one shot). Does not alter the loss or its gradient. See losses.py.
    lm_chunk_rows: int = 4096


@dataclass
class OptimConfig:
    name: str = "muon"           # muon | adamw
    lr: float = 2e-2             # base LR; each rule scales it by lr_mult
    min_lr_ratio: float = 0.1    # cosine floor = lr * min_lr_ratio
    warmup_steps: int = 100
    grad_clip: float = 1.0
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    adamw_betas: List[float] = field(default_factory=lambda: [0.9, 0.95])


@dataclass
class ParallelConfig:
    """Degrees per mesh dimension. Only DP is implemented; the others are the seams
    where FSDP/EP/TP slot in, and asking for them fails loudly."""
    dp: int = -1                 # -1 = whatever is left of world_size
    ep: int = 1
    tp: int = 1
    compile: bool = False
    activation_checkpointing: bool = False
    mixed_precision: str = "bf16"   # bf16 | none (autocast on CUDA only)


@dataclass
class DataConfig:
    source: str = "stream"       # stream | disk | synthetic
    # Token cache for source=disk, built by `python -m moestack.prepare_data`.
    cache_dir: Optional[str] = None
    seq_len: int = 2048
    batch_size: int = 16         # micro-batch per rank
    grad_accum: int = 16
    num_workers: int = 4
    eval_batches: int = 10
    tokenizer: str = "cl100k"
    insert_separator: str = "before"


@dataclass
class TrainConfig:
    n_steps: int = 2_000
    seed: int = 42
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 500
    save_dir: str = "checkpoints"
    resume: Optional[str] = None
    auto_resume: bool = False
    device: Optional[str] = None
    dry_run: bool = False        # 100 steps, eval at 50
    # A step whose loss or grad norm is non-finite is dropped rather than applied.
    # Isolated spikes are survivable; this many in a row means something is broken.
    max_consecutive_skips: int = 5


@dataclass
class LoggingConfig:
    wandb: bool = True
    project: str = "moe-baseline"
    entity: Optional[str] = None
    tag: Optional[str] = None
    group: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    load_stats: bool = True       # per-layer gini / dead-expert fraction


# Fields that cannot change what a run learns, and so must not change its name.
# Everything NOT listed here feeds the fingerprint -- which is the point: a new
# option is part of the run's identity by default, and has to be excluded on
# purpose. `train.seed` is excluded because run_name spells it out separately.
_MISSING = object()
_NAME_IGNORE_SECTIONS = ("logging",)
_NAME_IGNORE_FIELDS = {
    "train": ("save_dir", "resume", "auto_resume", "device", "log_every", "eval_every",
              "save_every", "dry_run", "max_consecutive_skips", "seed"),
    "data": ("num_workers", "eval_batches", "cache_dir"),
    "parallel": ("compile", "activation_checkpointing"),
    "loss": ("lm_chunk_rows",),
}


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        checks = [("optim.name", self.optim.name, ("muon", "adamw")),
                  ("parallel.mixed_precision", self.parallel.mixed_precision, ("bf16", "none")),
                  ("data.source", self.data.source, ("stream", "disk", "synthetic")),
                  ("model.experts_impl", self.model.experts_impl, ("grouped_mm", "loop"))]
        for key, val, allowed in checks:
            if val not in allowed:
                raise ValueError(f"{key} must be one of {allowed}, got {val!r}")
        if self.data.source == "disk" and not self.data.cache_dir:
            raise ValueError("data.source=disk needs data.cache_dir "
                             "(build one with `python -m moestack.prepare_data --out ...`)")

    def significant_overrides(self) -> dict:
        """{dotted key: value} for every learning-relevant field that differs from
        the dataclass defaults. This -- not a curated list -- is what identifies a
        run. Diffing against the defaults (rather than hashing the whole config)
        keeps names stable when a new field with a default is added to config.py.
        """
        cur, base = self.to_dict(), Config().to_dict()
        out = {}
        for section, fields in cur.items():
            if section in _NAME_IGNORE_SECTIONS:
                continue
            ignored = _NAME_IGNORE_FIELDS.get(section, ())
            for k, v in fields.items():
                if k not in ignored and v != base[section].get(k, _MISSING):
                    out[f"{section}.{k}"] = v
        return out

    def fingerprint(self) -> str:
        """Six hex characters standing for everything the readable part of the name
        leaves out -- batch size, grad accum, seq len, n_steps, widths, warmup, ...
        Two runs share a checkpoint prefix only if they really are the same run."""
        sig = self.significant_overrides()
        if not sig:
            return "base"
        blob = json.dumps(sig, sort_keys=True, default=str)
        return hashlib.blake2s(blob.encode(), digest_size=3).hexdigest()

    def condition_name(self) -> str:
        """run_name() without the seed or the batch tag: one value per experimental
        condition. Used as the wandb group, so repeated runs of a condition (other
        seeds, other batches) land in one group and wandb plots their mean and
        spread. Mirrors run_tracking.build_condition_name in the original stack.
        """
        def fmt(x):
            return f"{x:g}"
        return (f"{self.model.arch}-{self.model.scale}-{self.optim.name}"
                f"-aux{fmt(self.loss.aux_coef)}-z{fmt(self.loss.z_coef)}"
                f"-lr{fmt(self.optim.lr)}-{self.fingerprint()}")

    def run_name(self) -> str:
        """Encodes every field that distinguishes a baseline cell, so two runs can
        never share checkpoint filenames by accident."""
        name = f"{self.condition_name()}-s{self.train.seed}"
        if self.logging.tag:
            name += f"-{self.logging.tag}"
        return name

    def to_dict(self) -> dict:
        return asdict(self)
