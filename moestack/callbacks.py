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

import math
import time
from dataclasses import dataclass, field

import torch


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

def gini(x: torch.Tensor) -> float:
    x = x.float().flatten().sort().values
    n = x.numel()
    if n == 0 or x.sum() <= 0:
        return 0.0
    idx = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    return float(((2 * idx - n - 1) * x).sum() / (n * x.sum()))


class MoELoadStats(Callback):
    """Per-step routing balance from the exact global-batch counts:
    load/gini, load/dead_expert_fraction, load/max_over_mean (layer means), plus the
    worst layer. Under EP, max_over_mean is the straggler factor of the busiest rank."""

    def on_step_end(self, trainer, state):
        c = state.expert_counts
        if c is None:
            return
        c = c.float()
        ginis = [gini(row) for row in c]
        mom = (c.max(dim=1).values / c.mean(dim=1).clamp(min=1e-9))
        state.metrics.update({
            "load/gini": sum(ginis) / len(ginis),
            "load/gini_max_layer": max(ginis),
            "load/dead_expert_fraction": float((c == 0).float().mean()),
            "load/max_over_mean": float(mom.mean()),
        })


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
