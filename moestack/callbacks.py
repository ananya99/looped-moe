"""Observation (and, later, intervention) without touching the training loop.

The trainer fires a small set of events with a `StepState`. Anything that wants to
log, measure or change something between steps is a `Callback`. Callbacks can be
stateful; their `state_dict` goes into the checkpoint.

Events (in order within a step):
    on_train_start(trainer)
    on_before_optimizer_step(trainer, state)   grads are final (reduced + clipped)
    on_step_end(trainer, state)                params updated
    on_eval(trainer, state, metrics)
    on_train_end(trainer, state)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .expert_lr import ema_update, lr_scales_from_counts, scale_summary
from .stats import delta_norms, drift_metrics, gini, load_metrics, row_norms


@dataclass
class StepState:
    step: int
    lr: float = 0.0
    loss_terms: dict = field(default_factory=dict)      # name -> float, mean over global batch
    grad_norm: float = float("nan")
    expert_counts: torch.Tensor | None = None           # [L, E] int64, summed over global batch
    tokens_seen: int = 0
    metrics: dict = field(default_factory=dict)         # callbacks add keys for loggers


class Callback:
    def on_train_start(self, trainer): ...
    def on_before_optimizer_step(self, trainer, state: StepState): ...
    def on_step_end(self, trainer, state: StepState): ...
    def on_eval(self, trainer, state: StepState, metrics: dict): ...
    def on_train_end(self, trainer, state: StepState): ...
    def state_dict(self) -> dict: return {}
    def load_state_dict(self, sd: dict): ...


class CallbackList(Callback):
    def __init__(self, callbacks: list[Callback]):
        self.callbacks = callbacks

    def _fire(self, name, *args):
        for cb in self.callbacks:
            getattr(cb, name)(*args)

    def on_train_start(self, t): self._fire("on_train_start", t)
    def on_before_optimizer_step(self, t, s): self._fire("on_before_optimizer_step", t, s)
    def on_step_end(self, t, s): self._fire("on_step_end", t, s)
    def on_eval(self, t, s, m): self._fire("on_eval", t, s, m)
    def on_train_end(self, t, s): self._fire("on_train_end", t, s)

    def state_dict(self):
        return {type(cb).__name__: cb.state_dict() for cb in self.callbacks}

    def load_state_dict(self, sd):
        for cb in self.callbacks:
            if type(cb).__name__ in sd:
                cb.load_state_dict(sd[type(cb).__name__])


# ──────────────────────────────────────────────────────────────────────
# Standard callbacks
# ──────────────────────────────────────────────────────────────────────

class MoELoadStats(Callback):
    """Per-step routing balance from the exact global-batch counts (see
    `stats.load_metrics`): gini, gini_alive, normalised entropy, dead-expert
    fraction, max/mean, each averaged over layers plus the worst layer.

    `per_layer` additionally emits one series per layer, which is 4 x n_layers
    series -- fine for wandb at these depths, and imbalance is very uneven across
    depth. Under EP, max_over_mean is the straggler factor of the busiest rank.
    """

    def __init__(self, per_layer: bool = True):
        self.per_layer = per_layer

    def on_step_end(self, trainer, state):
        if state.expert_counts is None:
            return
        state.metrics.update(load_metrics(state.expert_counts, per_layer=self.per_layer))


class PerExpertLR(Callback):
    """Per-expert learning rates, derived from this step's routing.

    The whole intervention is: read the exact global-batch expert counts the
    trainer already computed, turn them into one multiplier per expert
    (`expert_lr.lr_scales_from_counts`), and write them onto the Muon param groups
    (`Optimizers.set_expert_lr_scales`). No change to the loop, the model or the
    gradients -- which is the point: Muon's Newton-Schulz step normalises the
    update's singular values, so scaling a gradient does nothing and only the
    learning rate survives.

    Two things the original trainer had to arrange by hand are free here:

    - `state.expert_counts` is already summed over every micro-batch and
      all-reduced across ranks before any callback runs, so every replica derives
      identical multipliers. (Diverging per-rank LRs would silently desynchronise
      the replicas -- nothing would crash.)
    - The counts come from `RouterOut`, not forward hooks, so activation
      checkpointing replaying the forward inside backward cannot double them.

    The EMA is checkpointed, so a resumed run continues with a warm average rather
    than re-seeding from one step's counts.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.ema = None

    def on_train_start(self, trainer):
        if not trainer.optimizers.expert_lr_groups():
            # Config validation catches optim.name=adamw; this catches a Trainer
            # that built its optimizers without split_experts=True.
            raise RuntimeError("PerExpertLR: no per-layer expert param groups -- "
                               "build_optimizers(..., split_experts=True) is required")
        if trainer.verbose:
            c = self.cfg
            print(f"  per-expert LR: alpha={c.alpha} freeze_dead={c.freeze_dead} "
                  f"ema={c.ema} normalize={c.normalize} "
                  f"clamp=[{c.clamp_min}, {c.clamp_max}]")

    def on_before_optimizer_step(self, trainer, state):
        counts = state.expert_counts
        if counts is None:
            return
        c = self.cfg
        # Smoothed counts drive the share (per-step counts are very noisy exactly
        # where n_e is small); deadness stays on this step's raw counts, or an
        # expert with EMA history could never be recognised as dead.
        if c.ema > 0:
            self.ema = ema_update(self.ema, counts, c.ema)
            driving = self.ema
        else:
            driving = counts.float()
        scales = lr_scales_from_counts(driving, alpha=c.alpha, freeze_dead=c.freeze_dead,
                                       normalize=c.normalize, clamp_min=c.clamp_min,
                                       clamp_max=c.clamp_max, dead_counts=counts)
        trainer.optimizers.set_expert_lr_scales(scales)
        # The base rate an expert group would run at without the rule: the
        # scheduled LR times that group's lr_mult (1.0 for experts, but read it
        # rather than assume it -- a future rule may move experts off the full LR).
        group = next(iter(trainer.optimizers.expert_lr_groups().values()))[0]
        state.metrics.update(scale_summary(scales, counts,
                                           base_lr=state.lr * group["lr_mult"]))

    def state_dict(self):
        return {"ema": None if self.ema is None else self.ema.cpu()}

    def load_state_dict(self, sd):
        self.ema = sd.get("ema")


class ExpertStats(Callback):
    """How far each expert actually moved, next to how many tokens it saw.

    Per-expert series are far too wide for wandb (64 experts x 16 layers = 1024
    series per quantity), so the dashboard gets the correlations and the full joint
    distribution goes to `<save_dir>/<run_name>_expert_stats.jsonl`, one record per
    measured step. That file is what an offline analysis of "does token count
    predict learning" is computed from.

    The measurement brackets the optimizer step: a snapshot of the expert tensors
    at `on_before_optimizer_step`, the difference at `on_step_end`. Snapshots go to
    host memory -- at `regular` scale the expert weights are a sizeable fraction of
    a GPU, and this runs on a cadence precisely because it is not free.

    Steps the trainer *drops* (non-finite loss or grad norm) are skipped rather
    than recorded: no optimizer step ran, so every drift would be exactly zero and
    the correlation would be meaningless rather than absent.
    """

    def __init__(self, every: int, path=None):
        self.every, self.path = every, path
        self._before = None
        self._fh = None

    # -- plumbing ------------------------------------------------------
    def _experts(self, trainer):
        return [(m.experts.gate_up_proj, m.experts.down_proj)
                for m in trainer.model.moe_layers()]

    def _due(self, step):
        return self.every > 0 and step % self.every == 0

    def on_train_start(self, trainer):
        if self.every > 0 and self.path is None:
            self.path = Path(trainer.save_dir) / f"{trainer.run_name}_expert_stats.jsonl"

    def on_before_optimizer_step(self, trainer, state):
        if not self._due(state.step):
            return
        # .to("cpu") also detaches the copy from the GPU allocator, so the peak is
        # one transfer rather than a second resident copy of the expert weights.
        self._before = [(g.detach().to("cpu", copy=True), d.detach().to("cpu", copy=True))
                        for g, d in self._experts(trainer)]

    def on_step_end(self, trainer, state):
        before, self._before = self._before, None
        if before is None or state.expert_counts is None:
            return
        if not (math.isfinite(state.grad_norm)
                and math.isfinite(state.metrics.get("train/loss", float("nan")))):
            return                      # the trainer dropped this step; nothing moved
        after = [(g.detach().to("cpu", copy=True), d.detach().to("cpu", copy=True))
                 for g, d in self._experts(trainer)]
        d_gate = torch.stack([delta_norms(b[0], a[0]) for b, a in zip(before, after)])
        d_down = torch.stack([delta_norms(b[1], a[1]) for b, a in zip(before, after)])
        # One number per expert: the norm of its whole step, both matrices together.
        drift = (d_gate.square() + d_down.square()).sqrt()
        counts = state.expert_counts.cpu()
        state.metrics.update(drift_metrics(counts, drift))

        if trainer.dist.is_main and self.path is not None:
            scales = [g["lr_scale"] for layer, gs in
                      sorted(trainer.optimizers.expert_lr_groups().items()) for g in gs[:1]]
            self._write({
                "step": state.step,
                "lr": state.lr,
                "counts": counts.tolist(),
                "drift": drift.tolist(),
                "drift_gate_up": d_gate.tolist(),
                "drift_down": d_down.tolist(),
                "weight_norm_gate_up": torch.stack([row_norms(a[0]) for a in after]).tolist(),
                "weight_norm_down": torch.stack([row_norms(a[1]) for a in after]).tolist(),
                "lr_scale": [s.tolist() for s in scales] if scales and scales[0] is not None
                            else None,
            })

    def _write(self, record: dict):
        if self._fh is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = Path(self.path).open("a")
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()        # a killed job keeps every record it already measured

    def on_train_end(self, trainer, state):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class ConsoleLogger(Callback):
    def __init__(self, every: int):
        self.every = every
        self._t0 = None
        self._tokens0 = 0

    def on_train_start(self, trainer):
        self._t0, self._tokens0 = time.time(), trainer.tokens_seen

    def on_step_end(self, trainer, state):
        if not trainer.dist.is_main or (state.step % self.every and state.step >= 10):
            return
        dt = max(time.time() - self._t0, 1e-9)
        tps = (state.tokens_seen - self._tokens0) / dt
        state.metrics["train/tokens_per_sec"] = tps
        terms = " ".join(f"{k}={v:.4f}" for k, v in state.loss_terms.items())
        load = ""
        if "load/gini" in state.metrics:
            load = (f" gini={state.metrics['load/gini']:.3f}"
                    f" dead={state.metrics['load/dead_expert_fraction']:.3f}")
        print(f"[step {state.step}] {terms} gnorm={state.grad_norm:.3f} lr={state.lr:.2e} "
              f"tps={tps:,.0f}{load}", flush=True)

    def on_eval(self, trainer, state, metrics):
        if trainer.dist.is_main:
            print(f"[eval after {trainer.step} steps] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
                  flush=True)


class WandbLogger(Callback):
    """Logs every step's loss terms + whatever other callbacks put in state.metrics.
    Run id is checkpointed so a resumed job continues the same wandb run."""

    def __init__(self, cfg, run_name: str, config_dict: dict, group: str | None = None):
        self.cfg, self.run_name, self.config_dict = cfg, run_name, config_dict
        # An explicit logging.group wins; otherwise group by experimental condition.
        self.group = cfg.group or group
        self.run, self.run_id = None, None

    def on_train_start(self, trainer):
        if not trainer.dist.is_main:
            return
        import wandb
        self.run = wandb.init(project=self.cfg.project, entity=self.cfg.entity, name=self.run_name,
                              group=self.group, tags=self.cfg.tags, notes=self.cfg.notes,
                              config=self.config_dict, id=self.run_id,
                              resume="allow" if self.run_id else None)
        self.run_id = self.run.id

    def on_step_end(self, trainer, state):
        if self.run is None:
            return
        log = {f"train/{k}_loss" if k != "lm" else "train/lm_loss": v for k, v in state.loss_terms.items()}
        log.update({"train/loss": state.metrics.get("train/loss", float("nan")),
                    "train/grad_norm": state.grad_norm, "train/lr": state.lr,
                    "train/tokens_seen": state.tokens_seen})
        log.update(state.metrics)
        if torch.cuda.is_available():
            log["system/gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        self.run.log({k: v for k, v in log.items() if not (isinstance(v, float) and math.isnan(v))},
                     step=state.step)

    def on_eval(self, trainer, state, metrics):
        if self.run is not None:
            self.run.log({f"eval/{k}": v for k, v in metrics.items()}, step=state.step)

    def on_train_end(self, trainer, state):
        if self.run is not None:
            self.run.finish()

    def state_dict(self):
        return {"run_id": self.run_id}

    def load_state_dict(self, sd):
        self.run_id = sd.get("run_id")
