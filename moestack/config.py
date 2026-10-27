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
class PerExpertLRConfig:
    """Per-expert learning rates: `lr_e = base_lr * clamp(s_e ** alpha)`.

    Defaults are the baseline -- `alpha=0` with `freeze_dead=False` makes every
    multiplier exactly 1.0, which `expert_lr.py` and `optim/muon.py` turn into the
    unscaled code path, so a default run is bit-identical to one built before this
    option existed. See moestack/expert_lr.py for what each field does.
    """
    alpha: float = 0.0           # 0 = uniform; 0.5 / 1 = favour hot experts; <0 = cold
    freeze_dead: bool = False    # zero-token experts take no step and no weight decay
    # Counts driving the rule are smoothed over ~1/(1-ema) steps; deadness stays on
    # the raw per-step counts either way. 0.9 is what imbalance_train.py defaulted
    # to, i.e. what the batch-1 runs in RESULTS.md actually used.
    ema: float = 0.9
    normalize: bool = True       # rescale to mean 1.0 over the experts that got tokens
    clamp_min: float = 1e-3
    clamp_max: float = 10.0

    @property
    def enabled(self) -> bool:
        """Nothing is computed or applied unless a knob is actually turned."""
        return self.alpha != 0.0 or self.freeze_dead


@dataclass
class ExpertAccumConfig:
    """Expert-local gradient accumulation: each expert updates once it has been
    routed about `threshold` tokens, instead of once per optimizer step.

    Defaults are off, and `enabled=False` leaves every code path exactly as it was.
    See moestack/expert_accum.py for the rule and what each field costs.
    """
    enabled: bool = False
    # Which tick drives the decision. "step" is the batch-level version: experts
    # still update at step boundaries, just not every step. "microbatch" is the
    # event-driven one, where an expert can update several times within a step.
    # Plain str, not Literal: omegaconf cannot wrap a Literal annotation, and
    # the value is validated in Config.__post_init__ like the other enums.
    clock: str = "step"
    threshold: Optional[int] = None       # Q, in routed tokens
    # Q as a multiple of a balanced expert's per-tick load. The setting that
    # transfers across scales: at tiny scale a balanced expert already sees ~3k
    # tokens per micro-batch and ~49k per step.
    threshold_rel: Optional[float] = None
    max_window: Optional[int] = None      # D, the staleness cap, in ticks
    # "mean" divides an event's gradient by the tokens it was built from, so window
    # length does not decide how heavily an event weighs in the momentum buffer.
    grad_scale: str = "mean"


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
    per_expert_lr: PerExpertLRConfig = field(default_factory=PerExpertLRConfig)
    expert_accum: ExpertAccumConfig = field(default_factory=ExpertAccumConfig)


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
    # How many `_step*.pt` files to keep, newest first (0 = keep every one, the
    # original behaviour). A full-state checkpoint is the model plus the optimizer
    # state -- ~59 GB at `regular` scale -- so a run that saves often enough to
    # survive preemption fills a scratch quota long before it finishes. `best` is
    # never pruned, and neither is the newest step file, which is what
    # `auto_resume` loads.
    keep_last_n: int = 0
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
    load_stats: bool = True       # load/gini + load/dead_expert_fraction
    # Steps between per-expert records in <save_dir>/<run_name>_expert_stats.jsonl
    # (counts, per-expert ||dw||, weight norms, LR multipliers) and the drift/* and
    # update/rel metrics derived from them. 0 = off. Measurement cadence only: it is
    # in the `logging` section, so it never changes a run's name or fingerprint.
    #
    # Non-zero by default, which it did not use to be: the dashboard deliberately no
    # longer carries the per-layer and order-statistic aggregates (see
    # `stats.load_metrics`), so this file is now the only record of the `[L, E]`
    # matrices they were computed from. Setting it to 0 does not just skip a
    # measurement any more -- it throws the counts away.
    expert_stats_every: int = 50
    # Steps between records in <save_dir>/<run_name>_accum_events.jsonl: for each
    # tick on which something fired, the full [L, E] picture of which experts
    # closed a window, on how many tokens, by which rule, and what is still
    # waiting. The dashboard gets only the accum/* aggregates. 0 = off, but the
    # four accum/* series are now all the dashboard has, so leave it on.
    accum_events_every: int = 50
    # Steps between per-expert gradient-quality measurements (grad/snr,
    # grad/expert_norm). 0 = off. Needs `expert_stats_every` on the same cadence to
    # reach the JSONL, since that is the writer. The hooks are armed only on
    # measured steps, so a large cadence really is free on the others.
    grad_stats_every: int = 50


# Fields that cannot change what a run learns, and so must not change its name.
# Everything NOT listed here feeds the fingerprint -- which is the point: a new
# option is part of the run's identity by default, and has to be excluded on
# purpose. `train.seed` is excluded because run_name spells it out separately.
_MISSING = object()
_NAME_IGNORE_SECTIONS = ("logging",)
_NAME_IGNORE_FIELDS = {
    "train": ("save_dir", "resume", "auto_resume", "device", "log_every", "eval_every",
              "save_every", "keep_last_n", "dry_run", "max_consecutive_skips", "seed"),
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
        pe = self.optim.per_expert_lr
        if pe.enabled and self.optim.name != "muon":
            # Under AdamW the experts are single 3D tensors with one scalar lr, so
            # the rule has nothing to address. Fail at startup, not after the first
            # step, and never silently no-op.
            raise ValueError(f"optim.per_expert_lr needs optim.name=muon, got "
                             f"{self.optim.name!r}: per-expert multipliers are applied "
                             f"inside Muon's batched step")
        if not 0.0 <= pe.ema < 1.0:
            raise ValueError(f"optim.per_expert_lr.ema must be in [0, 1), got {pe.ema}")
        if not 0.0 <= pe.clamp_min <= pe.clamp_max:
            raise ValueError(f"optim.per_expert_lr needs 0 <= clamp_min <= clamp_max, "
                             f"got [{pe.clamp_min}, {pe.clamp_max}]")
        ea = self.optim.expert_accum
        if ea.enabled:
            if self.optim.name != "muon":
                # Same reason as per_expert_lr: under AdamW the experts are single
                # 3D tensors with one step for all of them, so there is nothing to
                # fire individually.
                raise ValueError(f"optim.expert_accum needs optim.name=muon, got "
                                 f"{self.optim.name!r}: an expert is addressed one "
                                 f"slice at a time inside Muon's batched step")
            if pe.enabled:
                # They compose in principle (mask x multiplier), but the LR rule is
                # derived per step while accumulation's natural unit is per event.
                # Running both would confound two interventions in one cell.
                raise ValueError("optim.expert_accum and optim.per_expert_lr cannot "
                                 "both be on: they are separate interventions and "
                                 "would confound each other")
            if ea.clock not in ("step", "microbatch"):
                raise ValueError(f"optim.expert_accum.clock must be step | microbatch, "
                                 f"got {ea.clock!r}")
            if ea.grad_scale not in ("mean", "sum"):
                raise ValueError(f"optim.expert_accum.grad_scale must be mean | sum, "
                                 f"got {ea.grad_scale!r}")
            if ea.threshold is not None and ea.threshold_rel is not None:
                raise ValueError("optim.expert_accum takes at most one of threshold "
                                 "and threshold_rel")
            if ea.threshold is None and ea.threshold_rel is None and ea.max_window is None:
                raise ValueError("optim.expert_accum needs a threshold, a threshold_rel "
                                 "or a max_window -- with none of them, no expert "
                                 "would ever update")
            if ea.max_window is not None and ea.max_window < 1:
                raise ValueError(f"optim.expert_accum.max_window must be at least 1 "
                                 f"tick, got {ea.max_window}")
            # Resume is allowed. An open window lives in `p.grad`, which
            # checkpoint.py does not write -- storing it would add the whole expert
            # gradient to every file and make the save slower, which is the opposite
            # of what a preempted run needs. So ExpertAccumulation discards the open
            # windows on load instead (the same move it makes for a non-finite
            # batch) and reports it as `accum/resumes`. Event history survives; at
            # most `max_window` ticks of partial gradient per expert do not.

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
                if k in ignored or v == base[section].get(k, _MISSING):
                    continue
                if isinstance(v, dict):
                    # Nested config block (optim.per_expert_lr): report the fields
                    # that differ, not the whole dict, so the printed override line
                    # and the fingerprint both stay field-granular.
                    sub_base = base[section].get(k) or {}
                    out.update({f"{section}.{k}.{sk}": sv for sk, sv in v.items()
                                if sv != sub_base.get(sk, _MISSING)})
                else:
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
        pe = self.optim.per_expert_lr
        # "aneg0.5" rather than "a-0.5": a bare minus reads as a separator in both
        # the filename and the wandb run name (same convention as run_tracking.py).
        per_expert = ""
        if pe.alpha != 0.0:
            per_expert += f"-aneg{fmt(abs(pe.alpha))}" if pe.alpha < 0 else f"-a{fmt(pe.alpha)}"
        if pe.freeze_dead:
            per_expert += "-freeze"
        ea, accum = self.optim.expert_accum, ""
        if ea.enabled:
            budget = ("q" + fmt(ea.threshold) if ea.threshold is not None else
                      "r" + fmt(ea.threshold_rel) if ea.threshold_rel is not None else
                      "fixed")
            accum = f"-acc{'mb' if ea.clock == 'microbatch' else 'st'}{budget}"
            if ea.max_window:
                accum += f"w{ea.max_window}"
            if ea.grad_scale != "mean":
                accum += f"-{ea.grad_scale}"
        return (f"{self.model.arch}-{self.model.scale}-{self.optim.name}"
                f"-aux{fmt(self.loss.aux_coef)}-z{fmt(self.loss.z_coef)}"
                f"-lr{fmt(self.optim.lr)}{per_expert}{accum}-{self.fingerprint()}")

    def run_name(self) -> str:
        """Encodes every field that distinguishes a baseline cell, so two runs can
        never share checkpoint filenames by accident."""
        name = f"{self.condition_name()}-s{self.train.seed}"
        if self.logging.tag:
            name += f"-{self.logging.tag}"
        return name

    def to_dict(self) -> dict:
        return asdict(self)
