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
from .compass import expert_scale
from .grad_stats import ExpertGradMoments, accumulated_grad_sq
from .stats import (delta_norms, drift_metrics, gini, load_metrics, row_norms,
                    summarize)


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

    # The events on which callbacks write `state.metrics`. `on_microbatch_end`
    # writes none and fires `grad_accum` times a step, so it is left unguarded.
    _GUARDED = frozenset({"on_before_optimizer_step", "on_accepted_step", "on_step_end"})

    def _fire(self, name, *args):
        # Two writers of one metric key in a step is always a bug: the later one
        # wins silently and the series alternates between two different
        # quantities, which looks like a jump in the data rather than a defect.
        # `update/rel` was exactly this -- an interval measurement from
        # `ExpertStats` and a per-event one from `ExpertGradStats` under one name,
        # with the collision visible only as a step-50 discontinuity. Caught here,
        # on the first reporting step, rather than in the analysis months later.
        guard = name in self._GUARDED
        state = args[1] if guard else None
        for cb in self.callbacks:
            before = dict(state.metrics) if guard else None
            getattr(cb, name)(*args)
            if not guard:
                continue
            clash = sorted(k for k, v in before.items()
                           if k in state.metrics and state.metrics[k] is not v)
            if clash:
                raise RuntimeError(
                    f"{type(cb).__name__}.{name} overwrote {clash}, already written "
                    f"earlier in step {state.step}. Two publishers are sharing a "
                    f"metric key; give one of them its own name.")

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
    """Per-step routing balance from the exact global-batch counts: `load/gini` and
    `load/dead_expert_fraction` (see `stats.load_metrics` for why only those two).

    Imbalance really is uneven across depth, but the per-layer breakdown belongs in
    the `[L, E]` counts that `ExpertStats` writes to the JSONL, not in 4 x n_layers
    panels on every run's dashboard. Under EP, note that a per-layer max/mean
    recomputed from those counts is the straggler factor of the busiest rank.
    """

    def on_step_end(self, trainer, state):
        if state.expert_counts is None:
            return
        state.metrics.update(load_metrics(state.expert_counts))


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
        # Absolute per-expert rates are not logged: they are these multipliers times
        # the scheduled LR, which is already on the dashboard as `train/lr`.
        state.metrics.update(scale_summary(scales, counts,
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

    **Resumable, by discarding.** The window *counters* are checkpointed; the
    gradient they stand for is not, because it lives in `p.grad` and writing it
    would add the entire expert gradient to every checkpoint -- making the save
    slower, which is exactly backwards for a run that is being preempted. So a
    resumed run restores the event history and drops every open window, the same
    move `discard()` makes for a non-finite batch. The cost is bounded: at most
    `max_window` ticks of partial gradient per expert, once per resume, counted as
    `accum/resumes` so it is visible rather than silent. An uninterrupted run is
    unaffected -- nothing here runs unless a checkpoint was loaded.
    """

    def __init__(self, cfg, every: int = 0, path=None):
        self.cfg = cfg
        self.state, self.threshold, self.experts = None, None, []
        self._last_tokens = None
        self._fired, self._accepted, self._off = None, False, None
        self.tally = None
        self.discarded, self.nonfinite_micro = 0, 0
        # Resume bookkeeping. `_resumed` is the checkpointed state waiting to be
        # applied: load_state_dict runs in Trainer.__init__, before on_train_start
        # has built self.state or found the expert tensors, so it can only stash.
        self._resumed, self.resumes, self.resume_dropped = None, 0, 0
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
        self._restore(trainer)
        self._off = torch.zeros_like(self.state.pending, dtype=torch.bool)
        self.tally = EventTally(args.n_layers * args.n_experts)
        per_tick = trainer.tokens_per_step
        if self.cfg.clock == "microbatch":
            per_tick //= trainer.cfg.data.grad_accum
        # `threshold_rel` resolves against the per-STEP balanced load on **both**
        # clocks, not the per-tick one. A tick is a micro-batch on the mb clock and
        # a step on the st clock, so a per-tick reading made `threshold_rel=4` two
        # different experiments depending on the clock -- which is why the mb cells
        # used to pass an absolute token count instead, and why their run names read
        # `q49152` while the paired st cells read `r4`. Per-step units make the pair
        # name-identical and the conversion unnecessary. No-op for the st clock,
        # where a tick already is a step.
        self.threshold = resolve_threshold(self.cfg.threshold, self.cfg.threshold_rel,
                                           trainer.tokens_per_step,
                                           args.top_k, args.n_experts)
        if trainer.verbose:
            balanced = per_tick * args.top_k / args.n_experts
            print(f"  expert accumulation: clock={self.cfg.clock} Q={self.threshold} "
                  f"D={self.cfg.max_window} grad_scale={self.cfg.grad_scale} "
                  f"({balanced:,.0f} tokens/expert/tick at perfect balance)")

    def _restore(self, trainer):
        """Apply a checkpointed state, dropping whatever windows were open.

        `events` and `last_event` survive -- the record of what actually updated is
        still true, and `accum/event_gini` would otherwise restart from zero at
        every preemption and read as a step change in the metric. `pending` and
        `age` do not: their gradient is not in the checkpoint, and carrying the
        counts without it would scale the next event's gradient by tokens that are
        no longer there.
        """
        if self._resumed is None:
            return
        sd = self._resumed
        self._resumed = None
        self.state.load_state_dict(sd["accum"])
        self.discarded = sd.get("discarded", 0)
        self.nonfinite_micro = sd.get("nonfinite_micro", 0)
        self.resumes = sd.get("resumes", 0) + 1
        dropped = int((self.state.pending > 0).sum())
        self.resume_dropped = sd.get("resume_dropped", 0) + dropped
        self.state.discard()
        if trainer.verbose:
            print(f"  expert accumulation: resumed (#{self.resumes}); dropped "
                  f"{dropped} open window(s), kept the event history")

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
        # `_close_windows` already opened the event (it has to, before `_arm`
        # rescales the gradients). Closing it after the step is what captures the
        # displacement -- the only point at which it is observable under this
        # clock, since by the end of the step `on_accepted_step` has masked the
        # experts off and a step-level before/after sees exactly zero.
        # Only the experts: the shared parameters are still filling their own
        # window and must keep the gradient they have accumulated so far.
        trainer.optimizers.step(only="expert")
        sink = self._grad_sink(trainer)
        if sink is not None:
            sink.close_event(trainer, event.fire)
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
        # `_close_windows` opens the event; the trainer steps right after this
        # returns, and `on_step_end` closes it.
        self._fired = self._close_windows(trainer, state.expert_counts,
                                          state.step).fire

    def on_step_end(self, trainer, state):
        if not self._accepted:
            self._zero_every_window()
            self.state.discard()
            self.discarded += 1
        else:
            if self._fired is not None:
                sink = self._grad_sink(trainer)
                if sink is not None:
                    sink.close_event(trainer, self._fired)
                self._zero(self._fired)     # the step clock steps inside the trainer
                self._fired = None
            state.metrics.update(accum_metrics(self.state, self.tally))
            state.metrics["accum/discarded_windows"] = self.discarded
            if self.resumes:
                # How much of this run's history is a restart, and how much
                # accumulation it cost. A cell with many resumes is not directly
                # comparable to one with none.
                state.metrics["accum/resumes"] = self.resumes
                state.metrics["accum/resume_dropped_windows"] = self.resume_dropped
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
        sink = self._grad_sink(trainer)
        if sink is not None:
            sink.open_event(trainer, event.fire, tokens)     # before the rescale
        self._arm(trainer, event, tokens)
        self._record(trainer.step, 0, event, tokens, reason="flush")
        trainer.optimizers.step(only="expert")
        if sink is not None:
            sink.close_event(trainer, event.fire)
        self._zero(event.fire)
        trainer.optimizers.set_expert_update_mask(None)
        return True

    # -- plumbing ------------------------------------------------------
    def _grad_sink(self, trainer):
        """`ExpertGradStats`, if it is running. It owns the per-event record; this
        callback only says when an event begins and ends."""
        return next((c for c in trainer.callbacks.callbacks
                     if isinstance(c, ExpertGradStats) and c.moments is not None), None)

    def _close_windows(self, trainer, counts, step, tick=0):
        """observe -> decide -> consume -> arm the optimizer. Returns the event."""
        self.state.observe(counts)
        event = self.state.ready(self.threshold, self.cfg.max_window)
        tokens = self.state.consume(event.fire, step)
        self._last_tokens = tokens      # the window's token budget, for the sink
        self.tally.add(event, tokens)
        # Before `_arm`, which rescales the firing gradients in place by
        # 1/tokens. `ExpertGradStats` compares the accumulated gradient against the
        # raw per-micro-batch norms its hooks recorded, so reading `p.grad` after
        # the rescale divides one side of that ratio and not the other -- SNR then
        # comes out ~tokens times too small, and only under accumulation, which is
        # what made a baseline and a Q=1 step clock disagree on identical windows.
        sink = self._grad_sink(trainer)
        if sink is not None and event.any:
            sink.open_event(trainer, event.fire, tokens)
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

    # -- checkpointing -------------------------------------------------
    def state_dict(self):
        if self.state is None:      # never started; nothing to say
            return {}
        return {"accum": self.state.state_dict(), "discarded": self.discarded,
                "nonfinite_micro": self.nonfinite_micro, "resumes": self.resumes,
                "resume_dropped": self.resume_dropped}

    def load_state_dict(self, sd):
        # Stash only: this runs from Trainer.__init__, before on_train_start has
        # built self.state. `_restore` applies it once there is something to apply.
        self._resumed = sd or None


class ExpertStats(Callback):
    """How far each expert actually got, over a **fixed interval**, next to how many
    tokens it saw over the same interval.

    Brackets the reporting interval, not a single optimizer step: a snapshot at step
    N and another at step N+`every`. Three reasons, in order of importance:

    1. **It is correct under every clock.** Under `clock=microbatch` the experts
       move *inside* the micro-batch loop and `on_accepted_step` then masks them
       off, so a step-level before/after brackets an interval in which nothing
       happens and reports exactly zero. An interval diff does not care when inside
       it the movement happened.
    2. **No composition bias.** Every expert gets the same interval whether it fired
       once or fifty times, so the population is fixed and the series is directly
       comparable with a baseline run. Aggregating per-event displacements instead
       would weight hot experts by how often they fire -- which is the very thing an
       intervention changes.
    3. **It is the quantity that carries the signal.** Under Muon a single update is
       nearly a fixed size, so per-event displacement says little; what varies is the
       *rate*, and only an interval measurement integrates firing rate.

    This is **net displacement**, not path length: updates that partly cancel show
    less motion than aligned ones. That is the right number for "how far did it get"
    and the wrong one for "how much did it move" -- path length is the sum of
    `rel_update` over the expert's events, in the grad-events JSONL.

    Published as `drift/rel_*`. `ExpertGradStats` publishes the per-event version as
    `update/rel_*`; both are ||dw||/||w|| and they are not interchangeable, so the
    namespace carries the clock.

    Snapshots go to host memory, which is why this runs on a cadence at all.
    """

    def __init__(self, every: int, path=None):
        self.every, self.path = every, path
        self._before = None
        self._counts = None
        self._fh = None

    def _experts(self, trainer):
        return [(m.experts.gate_up_proj, m.experts.down_proj)
                for m in trainer.model.moe_layers()]

    def _due(self, step):
        return self.every > 0 and step % self.every == 0

    def _snapshot(self, trainer):
        # .to("cpu") also detaches the copy from the GPU allocator, so the peak is
        # one transfer rather than a second resident copy of the expert weights.
        return [(g.detach().to("cpu", copy=True), d.detach().to("cpu", copy=True))
                for g, d in self._experts(trainer)]

    def on_train_start(self, trainer):
        if self.every > 0 and self.path is None:
            self.path = Path(trainer.save_dir) / f"{trainer.run_name}_expert_stats.jsonl"

    def on_step_end(self, trainer, state):
        if self.every <= 0 or state.expert_counts is None:
            return
        # Token share over the same interval as the displacement, so the two sides
        # of `corr_share_vs_update` describe the same stretch of training. Pairing
        # one step's counts with a window that opened ten steps ago was the other
        # way this metric went wrong under accumulation.
        c = state.expert_counts.cpu()
        self._counts = c.clone() if self._counts is None else self._counts + c
        if not self._due(state.step):
            return
        before, self._before = self._before, self._snapshot(trainer)
        counts, self._counts = self._counts, None
        if before is None:
            return              # first reporting step: an interval needs two ends
        after = self._before
        d_gate = torch.stack([delta_norms(b[0], a[0]) for b, a in zip(before, after)])
        d_down = torch.stack([delta_norms(b[1], a[1]) for b, a in zip(before, after)])
        drift = (d_gate.square() + d_down.square()).sqrt()
        w_gate = torch.stack([row_norms(a[0]) for a in after])
        w_down = torch.stack([row_norms(a[1]) for a in after])
        w_norm = (w_gate.square() + w_down.square()).sqrt()
        state.metrics.update(drift_metrics(counts, drift, weights=w_norm))

        if trainer.dist.is_main and self.path is not None:
            scales = [g["lr_scale"] for layer, gs in
                      sorted(trainer.optimizers.expert_lr_groups().items()) for g in gs[:1]]
            self._write({
                "step": state.step, "lr": state.lr, "interval": self.every,
                "counts": counts.tolist(),
                "drift": drift.tolist(),
                "drift_gate_up": d_gate.tolist(),
                "drift_down": d_down.tolist(),
                "weight_norm_gate_up": w_gate.tolist(),
                "weight_norm_down": w_down.tolist(),
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


class ExpertGradStats(Callback):
    """Per-expert update quality on the **event** clock, reported on a cadence.

    `grad/snr` asks whether the micro-gradients that fed one update agreed with each
    other (see `grad_stats` for the identity). The window *is* the unit: pooling
    micro-gradients across several windows would mix in the gradient changing
    because the weights moved between them, which is gradient non-stationarity, not
    noise. So the measurement clock is forced and only the reporting clock is free.

    Two rules follow, and getting either wrong corrupts the metric silently:

    1. **The bookkeeping runs continuously.** Window boundaries belong to
       `ExpertAccumulation`, which fires, steps and zeroes gradients on every step
       whether or not this callback is reporting. If measurement skipped off-cadence
       steps, `sumsq` would retain micro-batches whose gradient had been zeroed away
       and the ratio would be between two different intervals.
    2. **Aggregate per expert first, then over experts.** At a reporting step the
       events that closed are a biased sample -- hot experts fire far more often --
       so pooling events directly would weight them fifty to one, and that weighting
       would move with any intervention that changes firing rates. Averaging within
       an expert first gives every expert one vote.

    An expert that fired zero times in the interval reports NaN, which `summarize`
    masks out. Read `grad/snr` next to `accum/events_per_expert`, which says how many
    events backed each mean.

    A baseline step is itself one window of `grad_accum` micro-batches, which is what
    anchors the SNR-versus-window-tokens curve at the un-accumulated end.
    """

    # field in the event record -> metric namespace it is published under.
    # `update/*` is the **event** clock throughout: `update/rel` is one event's
    # displacement and `update/per_token` is that divided by the tokens that built
    # it. The interval counterpart on the step clock is `drift/rel`, from
    # `ExpertStats` -- a different measurement, which is why it has its own name.
    FIELDS = {"snr": "grad/snr", "grad_norm": "grad/expert_norm",
              "rel_update": "update/rel", "per_token": "update/per_token"}

    def __init__(self, every: int, path=None):
        self.every, self.path = every, path
        self.moments = None
        self.last = None
        self._sum = self._n = None      # [L, E] running per-expert accumulators
        self._pending = None            # weights snapshot between open/close
        self._external = False
        self._fh = None

    def _due(self, step):
        return self.every > 0 and step % self.every == 0

    def _experts(self, trainer):
        return [(m.experts.gate_up_proj, m.experts.down_proj)
                for m in trainer.model.moe_layers()]

    def on_train_start(self, trainer):
        if self.every <= 0:
            return
        L, E = trainer.args.n_layers, trainer.args.n_experts
        self.moments = ExpertGradMoments(L, E, device=trainer.dist.device)
        self.moments.attach(trainer.model.moe_layers())
        self.moments.arm()              # permanently: see rule 1 above
        self._external = any(isinstance(c, ExpertAccumulation)
                             for c in trainer.callbacks.callbacks)
        self._fresh(L, E)
        if self.path is None:
            self.path = Path(trainer.save_dir) / f"{trainer.run_name}_grad_events.jsonl"

    def _fresh(self, L, E):
        self._sum = {k: torch.zeros(L, E) for k in self.FIELDS}
        self._n = {k: torch.zeros(L, E) for k in self.FIELDS}

    def on_microbatch_end(self, trainer, state, micro_step, counts, loss):
        # Registered before ExpertAccumulation, so this tick lands before the
        # controller closes any window that this micro-batch fed.
        if self.moments is not None:
            self.moments.tick()

    # -- the event -----------------------------------------------------
    def open_event(self, trainer, fire, tokens=None, grad_correction=1.0):
        """Called with the experts whose window just closed, *before* they step:
        the gradients are complete and the weights have not moved yet.

        `grad_correction` divides out anything that rescaled `p.grad` after the
        hooks saw the micro-gradients. SNR compares the accumulated gradient
        against the raw per-micro-batch norms, so a factor applied to one side and
        not the other is pure error -- and both of the things that do this are
        global, so neither carries information about within-window agreement.
        """
        if self.moments is None:
            return
        sel = fire.cpu().bool()
        grad_sq = accumulated_grad_sq(trainer.model.moe_layers())
        if grad_correction != 1.0:
            grad_sq = grad_sq * (grad_correction ** 2)      # grad_sq is a *squared* norm
        m = self.moments.close(grad_sq, world_size=trainer.dist.world_size)
        idx = sel.nonzero(as_tuple=False)
        rows = {(int(l), int(e)): None for l, e in idx}
        pairs = self._experts(trainer)
        for (l, e) in rows:
            g, d = pairs[l]
            rows[(l, e)] = (g.detach()[e].float().clone(),
                            d.detach()[e].float().clone(),
                            float(g.detach()[e].float().norm() ** 2
                                  + d.detach()[e].float().norm() ** 2) ** 0.5)
        self._pending = (sel, m, rows, None if tokens is None else tokens.cpu())

    def close_event(self, trainer, fire):
        """Called after those experts stepped: completes the record and restarts
        their gradient windows."""
        if self.moments is None or self._pending is None:
            return
        sel, m, rows, tokens = self._pending
        self._pending = None
        pairs = self._experts(trainer)
        for (l, e), (g0, d0, w0) in rows.items():
            g, d = pairs[l]
            dw = float(((g.detach()[e].float() - g0).norm() ** 2
                        + (d.detach()[e].float() - d0).norm() ** 2) ** 0.5)
            rel = dw / w0 if w0 > 0 else float("nan")
            tok = None if tokens is None else float(tokens[l, e])
            rec = {"snr": float(m["snr"][l, e]), "grad_norm": float(m["grad_norm"][l, e]),
                   "rel_update": rel,
                   # The quantity that makes windows of different length
                   # commensurable, and the thesis's actual question: does a bigger
                   # window buy a proportionally bigger step, or the same step in a
                   # better direction?
                   "per_token": (rel / tok if tok else float("nan"))}
            for k, v in rec.items():
                if v == v:                      # not NaN
                    self._sum[k][l, e] += v
                    self._n[k][l, e] += 1
            if self._fh is not None or self.path is not None:
                self._write({"step": trainer.step, "layer": l, "expert": e,
                             "tokens": tok, "micro": int(m["micro"][l, e]), **rec})
        self.moments.reset(fire)

    def _write(self, record):
        if self._fh is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = Path(self.path).open("a")
        self._fh.write(json.dumps(record) + "\n")

    # -- the baseline clock --------------------------------------------
    def on_before_optimizer_step(self, trainer, state):
        if self.moments is None or self._external:
            return
        # A baseline window *is* the step, so the step's counts are its token
        # budget. Supplying them is what makes `update/per_token` comparable
        # between a baseline cell and an accumulated one -- which is the whole
        # point of the metric, and it would silently be NaN here otherwise.
        #
        # The correction undoes gradient clipping. Only the baseline needs it:
        # `Trainer._clipped_params` deliberately excludes the expert tensors once
        # they are accumulators, so under accumulation nothing rescaled them, while
        # here `clip_grad_norm_` already shrank every expert gradient by a global
        # factor that the hooks never saw.
        clip = trainer.cfg.optim.grad_clip
        coef = 1.0
        if clip and math.isfinite(state.grad_norm):
            # Mirrors torch.nn.utils.clip_grad_norm_, which scales by
            # max_norm / (total_norm + 1e-6) and only when that is below 1.
            coef = min(1.0, clip / (state.grad_norm + 1e-6))
        self.open_event(trainer, torch.ones_like(self.moments.micro, dtype=torch.bool),
                        state.expert_counts,
                        grad_correction=(1.0 / coef) if coef > 0 else 1.0)

    def on_step_end(self, trainer, state):
        if self.moments is None:
            return
        if not self._external:
            self.close_event(trainer, torch.ones_like(self.moments.micro, dtype=torch.bool))
        if self._fh is not None:
            self._fh.flush()
        if not self._due(state.step):
            return
        # Stage one: per expert, the mean over its events in this interval.
        self.last = {k: torch.where(self._n[k] > 0, self._sum[k] / self._n[k].clamp(min=1),
                                    torch.full_like(self._sum[k], float("nan")))
                     for k in self.FIELDS}
        self.last["events"] = self._n["snr"].clone()
        for k, name in self.FIELDS.items():     # stage two: over experts
            v = self.last[k]
            state.metrics.update(summarize(v, name, mask=torch.isfinite(v)))
        self._fresh(*self._sum["snr"].shape)

    def on_train_end(self, trainer, state):
        if self.moments is not None:
            self.moments.detach()
            self.moments = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class CompassStats(Callback):
    """Compass alignment factors (arXiv:2610.04140), measured but **not applied**.

    Does the step Muon is about to take agree with the gradient it came from? Muon
    normalises the update's singular values, so a poorly-aligned expert still moves
    about as far as a well-aligned one -- and alignment degrades exactly when the
    tokens routed to an expert change while its momentum still reflects the old
    ones, which is this project's premise reached from the optimizer side.

    Pairs with `ExpertGradStats`: SNR asks whether an expert's gradient is
    trustworthy, alignment asks whether the update agrees with it. A cold expert can
    be bad on both counts for different reasons.

    `compass/scale` is `c_e * alpha_e`, the multiplier Compass *would* apply. It is
    only logged here, so no run's trajectory changes and this is safe to switch on
    mid-grid. The paper reports it spanning ~0.55-0.7 (smallest) to ~1.05-1.10
    (largest), with the mean rising toward 1.0 as the schedule decays.

    The measurement rides Muon's `probe` hook, which fires inside `_apply` where the
    orthogonalized direction exists; see `optim/muon.py`.
    """

    def __init__(self, every: int):
        self.every = every
        self.slot = {}          # id(param) -> (layer, which matrix)
        self.buf = None
        self.last = None
        self._lam = 1.0
        self._alive = None

    def _due(self, step):
        return self.every > 0 and step % self.every == 0

    def on_train_start(self, trainer):
        if self.every <= 0:
            return
        muon = trainer.optimizers.optimizers.get("muon")
        if muon is None:
            raise RuntimeError("CompassStats needs optim.name=muon: the factors are "
                               "cosines against Muon's orthogonalized direction, "
                               "which AdamW never forms")
        for li, moe in enumerate(trainer.model.moe_layers()):
            for wi, prm in enumerate((moe.experts.gate_up_proj, moe.experts.down_proj)):
                self.slot[id(prm)] = (li, wi)
        self._fresh(trainer.args.n_layers, trainer.args.n_experts)
        muon.probe = self._record

    def _fresh(self, L, E):
        nan = torch.full((L, E), float("nan"))
        # Two families per layer -- gate_up and down are separate families in the
        # paper, since a family is one projection matrix across the layer's experts.
        self.buf = {k: [nan.clone(), nan.clone()]
                    for k in ("alignment", "family", "radius", "scale")}

    def _record(self, p, idx, grad, update):
        """Muon's probe. Read-only: it must not touch `update`."""
        slot = self.slot.get(id(p))
        if slot is None or update is None or p.dim() != 3:
            return              # attention / router matrices are not expert families
        layer, which = slot
        alive = None if self._alive is None else (
            self._alive[layer] if idx is None else self._alive[layer][idx.cpu()])
        out = expert_scale(update, grad, self._lam, alive=alive)
        rows = slice(None) if idx is None else idx.cpu()
        for k, v in out.items():
            self.buf[k][which][layer, rows] = v.detach().cpu().float()

    def on_before_optimizer_step(self, trainer, state):
        # lam = lr / peak_lr, so the row factors relax to 1 as the schedule decays.
        self._lam = state.lr / max(trainer.cfg.optim.lr, 1e-12)
        self._alive = (None if state.expert_counts is None
                       else (state.expert_counts > 0).cpu())
        if not self._due(state.step):
            trainer.optimizers.optimizers["muon"].probe = None
        else:
            trainer.optimizers.optimizers["muon"].probe = self._record

    def on_step_end(self, trainer, state):
        if self.buf is None or not self._due(state.step):
            return
        self.last = {k: torch.stack(v) for k, v in self.buf.items()}   # [2, L, E]
        for k, v in self.last.items():
            # Over every expert *matrix*, as the paper does (its Fig. 3a spans "the
            # 96 expert matrices"), laid out [L, 2E] rather than [2L, E] so that
            # `argmax_layer` is a real layer index and `outlier_fraction` -- which
            # `summarize` judges per row -- is judged per layer. `argmax_expert` is
            # then `family * n_experts + expert`, family 0 = gate_up, 1 = down.
            flat = v.permute(1, 0, 2).reshape(v.size(1), -1)
            state.metrics.update(summarize(flat, f"compass/{k}",
                                           mask=torch.isfinite(flat)))
        self._fresh(trainer.args.n_layers, trainer.args.n_experts)

    def on_train_end(self, trainer, state):
        muon = trainer.optimizers.optimizers.get("muon")
        if muon is not None:
            muon.probe = None


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


# Numbers that are evidence a run is *valid* rather than curves to watch: monotone
# counters, constants, and fractions that are zero until something has gone wrong.
# They go to the wandb run summary -- one value per run, sortable in the runs table,
# no panel. Dropping them entirely would be the wrong trade: `resumes` is why a
# preempted cell is not directly comparable to a clean one, and `clamped_fraction`
# is why an alpha sweep may have stopped comparing equal-average cells. You need
# them to defend a result; you do not need to watch them arrive.
SUMMARY_ONLY = frozenset({
    "train/skipped_steps",
    "system/gpu_peak_gb",                 # already a running max, so last write wins
    "lr_scale/frozen_fraction",
    "lr_scale/clamped_fraction",
    "accum/threshold",                    # resolved once at train start; a constant
    "accum/discarded_windows",
    "accum/nonfinite_microbatches",
    "accum/resumes",
    "accum/resume_dropped_windows",
})


class WandbLogger(Callback):
    """Logs every step's loss terms + whatever other callbacks put in state.metrics.
    Run id is checkpointed so a resumed job continues the same wandb run.

    Keys in `SUMMARY_ONLY` are diverted to the run summary instead of being logged
    as series. Everything else a callback puts in `state.metrics` becomes a panel,
    which is the contract that keeps the dashboard honest: a metric is added here
    only when it names a decision, and the per-expert `[L, E]` matrices behind the
    aggregates live in the JSONLs for offline analysis."""

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
        summary = {k: log.pop(k) for k in list(log) if k in SUMMARY_ONLY}
        if summary:
            self.run.summary.update(summary)
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
