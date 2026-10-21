"""Observation (and, later, intervention) without touching the training loop.

The trainer fires a small set of events with a `StepState`. Anything that wants to
log, measure or change something between steps is a `Callback`. Callbacks can be
stateful; their `state_dict` goes into the checkpoint.

Events (in order within a step):
    on_train_start(trainer)
    on_microbatch_end(trainer, state, i, counts, loss)   after each backward
    on_before_optimizer_step(trainer, state)   grads are final (reduced + clipped)
    on_accepted_step(trainer, state)           ... and the step was not dropped
    on_step_end(trainer, state)                params updated
    on_eval(trainer, state, metrics)
    on_before_final_eval(trainer, state)       -> bool: "did you move the weights?"
    on_train_end(trainer, state)

`on_microbatch_end` carries that micro-batch's *rank-local* `[L, E]` counts and
its detached loss; only `state.expert_counts` is reduced over the global batch.
The loss is handed over rather than checked by the trainer because reading it
costs a device sync, and only a callback that acts mid-batch needs to pay for it. `on_accepted_step` is
`on_before_optimizer_step` minus the steps the trainer drops for a non-finite loss
or gradient norm -- anything that changes weights or consumes accumulated state
belongs there, because a dropped batch must leave both untouched.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .expert_accum import (AccumState, EventTally, accum_metrics, grad_scale,
                           resolve_threshold)
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
    def on_microbatch_end(self, trainer, state: StepState, micro_step: int,
                          counts: torch.Tensor, loss: torch.Tensor): ...
    def on_before_optimizer_step(self, trainer, state: StepState): ...
    def on_accepted_step(self, trainer, state: StepState): ...
    def on_step_end(self, trainer, state: StepState): ...
    def on_eval(self, trainer, state: StepState, metrics: dict): ...
    def on_before_final_eval(self, trainer, state: StepState) -> bool: return False
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
    def on_microbatch_end(self, t, s, i, c, l): self._fire("on_microbatch_end", t, s, i, c, l)
    def on_before_optimizer_step(self, t, s): self._fire("on_before_optimizer_step", t, s)
    def on_accepted_step(self, t, s): self._fire("on_accepted_step", t, s)
    def on_step_end(self, t, s): self._fire("on_step_end", t, s)
    def on_eval(self, t, s, m): self._fire("on_eval", t, s, m)
    def on_train_end(self, t, s): self._fire("on_train_end", t, s)

    def on_before_final_eval(self, t, s) -> bool:
        # The one event with a return value. A list, not a generator: every
        # callback has to run even once one of them has answered yes.
        return any([cb.on_before_final_eval(t, s) for cb in self.callbacks])

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

    `per_layer` additionally emits one series per layer under `load_per_layer/` --
    4 x n_layers, so 64 at depth 16. Off by default: imbalance really is uneven
    across depth, but 64 panels in every run's dashboard drown the eight
    aggregates. Under EP, max_over_mean is the straggler factor of the busiest rank.
    """

    def __init__(self, per_layer: bool = False):
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
        # Running sum of the multipliers, so the dashboard can show the
        # time-averaged rate each expert actually ran at. The instantaneous spread
        # bounces every step; what moves the weights is the integral, and an expert
        # that is hot early and cold later averages back to 1.0.
        self.cum, self.cum_n = None, 0

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
                                           base_lr=state.lr * group["lr_mult"],
                                           clamp_min=c.clamp_min, clamp_max=c.clamp_max))

        f = scales.detach().float()
        # .to(f): a resumed `cum` comes back on CPU (checkpoint.load maps to cpu,
        # and state_dict() stores it there), the same reason ema_update moves its
        # own carry-over. Without it the first step after a resume dies.
        self.cum = f.clone() if self.cum is None else self.cum.to(f) + f
        self.cum_n += 1
        alive = counts > 0
        if int(alive.sum()):
            m = (self.cum / self.cum_n)[alive]
            lo = float(m.min())
            state.metrics["lr_scale/cum_spread_alive"] = (
                float(m.max()) / lo if lo > 0 else float("nan"))

    def state_dict(self):
        return {"ema": None if self.ema is None else self.ema.cpu(),
                "cum": None if self.cum is None else self.cum.cpu(),
                "cum_n": self.cum_n}

    def load_state_dict(self, sd):
        self.ema = sd.get("ema")
        self.cum, self.cum_n = sd.get("cum"), sd.get("cum_n", 0)


class ExpertAccumulation(Callback):
    """Expert-local gradient accumulation: fix the token budget per update instead
    of the clock.

    The baseline gives every expert one update per step, built from whatever the
    router happened to send it -- a big-batch estimate for a hot expert, a handful
    of tokens for a cold one, and pure renormalised stale momentum for a dead one.
    Here an expert keeps its gradient until about `threshold` tokens have reached
    it, then takes one update and resets; hot experts simply update more often.

    The whole intervention is four lines of plumbing around a pure rule:

    1. `p.grad` *is* the window. Registering the expert tensors in
       `trainer.grad_accumulators` is what stops the trainer clearing them between
       batches and clipping them -- so accumulation costs no extra memory.
    2. `expert_accum.AccumState` decides who fires, from the `[L, E]` counts the
       model already returns. On the step clock those are all-reduced, so every
       rank decides identically with no extra collective.
    3. The firing slices' gradients are divided by the tokens they were built from
       and the rest are left alone, then `Optimizers.set_expert_update_mask` gates
       Muon so only those slices step -- weights, decay *and* momentum.
    4. Once the optimizer has stepped, those slices' gradients are zeroed; every
       other window carries on filling.

    Two clocks, one rule. `clock="step"` keeps updates at step boundaries and just
    skips the steps where a window is not full. `clock="microbatch"` lets an expert
    fire the moment its budget is met, several times within a step if it is hot --
    the global batch stops being the unit of time for experts, while the shared
    parameters keep stepping once per batch as usual. On that clock the shared
    gradient for one step is therefore computed against several versions of the
    experts; that is the trade the idea is making, and
    `accum/events_per_expert` is the number that says how far it is being pushed.

    Three cases that are easy to miss:

    - A batch the trainer **drops** as non-finite has already added its gradient to
      every open window, and it cannot be subtracted back out. So a dropped batch
      throws away every window rather than carrying a poisoned one -- which is also
      why the step-clock work happens on `on_accepted_step` and not on
      `on_before_optimizer_step`. On the micro-batch clock a non-finite micro-batch
      fires nothing, because an expert that has already stepped cannot be
      un-stepped; the step-level drop then clears the windows as usual.
    - The **final flush** closes whatever is still open before the run's last eval
      and save, or the last partial window of every expert is silently discarded.
      Only the experts step there; the shared parameters finished with the last
      batch.
    - **Momentum runs in event time.** An expert that fires every fiftieth tick
      decays its buffer once per event, not once per tick. That is the intended
      reading -- each expert has its own clock -- but it is a real difference from
      the baseline, not an implementation detail.

    Not resumable, deliberately: the open windows are in the gradient buffers,
    which `checkpoint.py` does not write. `config.py` refuses the combination
    rather than restoring token counts without the gradients they stand for.
    """

    def __init__(self, cfg, every: int = 0, path=None):
        self.cfg = cfg
        self.state, self.threshold, self.experts = None, None, []
        self._fired, self._accepted, self._off = None, False, None
        self.tally = None
        self.discarded, self.nonfinite_micro = 0, 0
        # Per-event records. Per-expert series are far too wide for wandb, so the
        # joint distribution goes to a file on a cadence and the dashboard gets
        # the aggregates -- the same split as ExpertStats.
        self.every, self.path, self._fh, self._main = every, path, None, True

    # -- setup ---------------------------------------------------------
    def on_train_start(self, trainer):
        if self.cfg.clock == "microbatch" and trainer.dist.world_size > 1:
            # The trainer runs every micro-batch but the last under `no_sync`, so
            # mid-batch an expert's gradient is this rank's alone and the replicas
            # would take different steps with nothing raising. Syncing every
            # micro-batch is correct (tests/test_ddp.py pins that) but costs
            # grad_accum times the communication, so it is a deliberate choice
            # rather than a silent default.
            raise NotImplementedError(
                f"optim.expert_accum.clock=microbatch is single-process for now, got "
                f"world_size={trainer.dist.world_size}: firing an expert mid-batch "
                f"needs its gradient reduced every micro-batch, not just at the "
                f"batch boundary. Use clock=step under DDP.")
        if not trainer.optimizers.expert_lr_groups():
            raise RuntimeError("ExpertAccumulation: no per-layer expert param groups -- "
                               "build_optimizers(..., split_experts=True) is required")
        args = trainer.args
        self._main = trainer.dist.is_main
        if self.every > 0 and self.path is None:
            self.path = Path(trainer.save_dir) / f"{trainer.run_name}_accum_events.jsonl"
        self.experts = [(m.experts.gate_up_proj, m.experts.down_proj)
                        for m in trainer.model.moe_layers()]
        trainer.grad_accumulators.extend(p for pair in self.experts for p in pair)
        self.state = AccumState.zeros(args.n_layers, args.n_experts,
                                      device=trainer.dist.device)
        self._off = torch.zeros_like(self.state.pending, dtype=torch.bool)
        self.tally = EventTally(args.n_layers * args.n_experts)
        per_tick = trainer.tokens_per_step
        if self.cfg.clock == "microbatch":
            per_tick //= trainer.cfg.data.grad_accum
        self.threshold = resolve_threshold(self.cfg.threshold, self.cfg.threshold_rel,
                                           per_tick, args.top_k, args.n_experts)
        if trainer.verbose:
            balanced = per_tick * args.top_k / args.n_experts
            print(f"  expert accumulation: clock={self.cfg.clock} Q={self.threshold} "
                  f"D={self.cfg.max_window} grad_scale={self.cfg.grad_scale} "
                  f"({balanced:,.0f} tokens/expert/tick at perfect balance)")

    # -- the cycle -----------------------------------------------------
    def on_microbatch_end(self, trainer, state, micro_step, counts, loss):
        if self.cfg.clock != "microbatch":
            return
        if not bool(torch.isfinite(loss)):
            # This micro-batch's gradient is already in every open window and
            # cannot be subtracted out, but an expert that has stepped cannot be
            # un-stepped either. Fire nothing; the step-level check drops the batch
            # and `on_step_end` discards the windows.
            self.nonfinite_micro += 1
            return
        event = self._close_windows(trainer, counts, state.step, tick=micro_step)
        if not event.any:
            return
        # Only the experts: the shared parameters are still filling their own
        # window and must keep the gradient they have accumulated so far.
        trainer.optimizers.step(only="expert")
        self._zero(event.fire)

    def on_before_optimizer_step(self, trainer, state):
        self._accepted = False      # until the trainer says otherwise

    def on_accepted_step(self, trainer, state):
        self._accepted = True
        if self.cfg.clock == "microbatch":
            # Every expert event for this batch already happened inside the loop.
            # Without an explicit all-off mask the optimizer would step them again,
            # on the zeroed gradient of whatever fired last.
            trainer.optimizers.set_expert_update_mask(self._off)
            return
        if state.expert_counts is None:
            return
        self._fired = self._close_windows(trainer, state.expert_counts,
                                          state.step).fire

    def on_step_end(self, trainer, state):
        if not self._accepted:
            self._zero_every_window()
            self.state.discard()
            self.discarded += 1
        else:
            if self._fired is not None:
                self._zero(self._fired)     # the step clock steps inside the trainer
                self._fired = None
            state.metrics.update(accum_metrics(self.state, self.tally, self.threshold))
            state.metrics["accum/discarded_windows"] = self.discarded
            if self.cfg.clock == "microbatch":
                state.metrics["accum/nonfinite_microbatches"] = self.nonfinite_micro
            if self.threshold:
                state.metrics["accum/threshold"] = self.threshold
        self.tally = EventTally(self.tally.slots)

    def on_before_final_eval(self, trainer, state) -> bool:
        event = self.state.flush()
        if not event.any:
            return False
        tokens = self.state.consume(event.fire, trainer.step)
        self.tally.add(event, tokens)
        self._arm(trainer, event, tokens)
        self._record(trainer.step, 0, event, tokens, reason="flush")
        trainer.optimizers.step(only="expert")
        self._zero(event.fire)
        trainer.optimizers.set_expert_update_mask(None)
        return True

    # -- plumbing ------------------------------------------------------
    def _close_windows(self, trainer, counts, step, tick=0):
        """observe -> decide -> consume -> arm the optimizer. Returns the event."""
        self.state.observe(counts)
        event = self.state.ready(self.threshold, self.cfg.max_window)
        tokens = self.state.consume(event.fire, step)
        self.tally.add(event, tokens)
        self._arm(trainer, event, tokens)
        self._record(step, tick, event, tokens)
        return event

    def _record(self, step, tick, event, tokens, reason="rule"):
        """One line per tick that fired anything, on the configured cadence."""
        if not (self._main and self.every > 0 and event.any
                and step % self.every == 0):
            return
        if self._fh is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = Path(self.path).open("a")
        self._fh.write(json.dumps({
            "step": step, "tick": tick, "clock": self.cfg.clock, "reason": reason,
            "threshold": self.threshold,
            # tokens is 0 wherever nothing fired, so it carries the mask too.
            "tokens": tokens.tolist(),
            "by_timeout": event.by_timeout.to(torch.int8).tolist(),
            "pending": self.state.pending.tolist(),
            "age": self.state.age.tolist(),
            "events": self.state.events.tolist(),
        }) + "\n")
        self._fh.flush()        # a killed job keeps every record it already wrote

    def on_train_end(self, trainer, state):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _arm(self, trainer, event, tokens):
        """Scale the firing gradients, and gate the optimizer to those slices."""
        if self.cfg.grad_scale != "sum":
            scales = grad_scale(tokens, event.fire, self.cfg.grad_scale)
            for layer, pair in enumerate(self.experts):
                # Only the firing slices are touched: a window that is still
                # filling must not be rescaled, and on the micro-batch clock this
                # runs grad_accum times per step.
                idx = event.fire[layer].nonzero().flatten()
                if not idx.numel():
                    continue
                v = scales[layer].index_select(0, idx).view(-1, 1, 1)
                for p in pair:
                    if p.grad is not None:
                        p.grad.index_copy_(0, idx, p.grad.index_select(0, idx) * v)
        trainer.optimizers.set_expert_update_mask(event.fire)

    def _zero(self, fire):
        for layer, pair in enumerate(self.experts):
            idx = fire[layer].nonzero().flatten()
            if not idx.numel():
                continue
            for p in pair:
                if p.grad is not None:
                    p.grad.index_fill_(0, idx, 0.0)

    def _zero_every_window(self):
        for pair in self.experts:
            for p in pair:
                if p.grad is not None:
                    p.grad.zero_()


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
