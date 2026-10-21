"""Expert-local gradient accumulation: token counts -> which experts step, when.

The baseline gives every expert one update per optimizer step, built from however
many tokens the router happened to send it. A hot expert's update is a big-batch
estimate; a cold expert's is a handful of tokens; a dead expert's is pure stale
momentum renormalised back to unit norm (RESULTS.md section 8). The idea here is
to fix the *token budget* instead of the clock: an expert accumulates gradient
until roughly `threshold` tokens have been routed to it, then takes one update and
resets. Hot experts simply update more often.

Two knobs define a window:

    threshold Q   tokens that close a window     (None = fixed-window variant)
    max_window D  ticks before it closes anyway  (the staleness cap)

A "tick" is whatever the caller counts: a global step for the batch-level clock,
a micro-batch for the event-driven one. Nothing in this file knows which.

Three rules that are easy to get wrong, and that the tests pin:

- **A window opens with its first token, not at the last reset.** An expert that
  receives nothing for fifty ticks and then one token has been waiting one tick,
  not fifty-one, so the timeout does not fire it immediately on arrival.
- **An expert with no tokens never fires, timeout or not.** Its gradient is exactly
  zero, and Newton-Schulz would renormalise its decaying momentum back to unit norm
  and take a full-magnitude step in a stale direction. "No tokens, no update" is
  the whole reason the mask in `optim/muon.py` freezes momentum too.
- **The gradient is divided by the tokens it was built from** (`grad_scale="mean"`),
  against the usual accumulation advice. Under Muon the overall scale washes out in
  Newton-Schulz, so this is not about step size: it is about the momentum buffer,
  which `buf.lerp_(grad, 1 - mom)` blends in proportion to the gradient's magnitude.
  Without the division a window spanning fifty ticks would dominate the buffer over
  one spanning two, purely because it is longer.

Everything here is a pure function of a `[L, E]` count matrix -- the same
`StepState.expert_counts` the load statistics and the per-expert LR rules read.
Nothing touches the model, the optimizer or `torch.distributed`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

GRAD_SCALES = ("mean", "sum")


# ──────────────────────────────────────────────────────────────────────
# Thresholds
# ──────────────────────────────────────────────────────────────────────

def mean_tokens_per_expert(tokens_per_tick: int, top_k: int, n_experts: int) -> float:
    """Tokens a perfectly balanced expert would receive per tick.

    `tokens_per_tick` is the global token count of one tick -- batch x seq_len x dp
    for a micro-batch clock, times grad_accum for a step clock. Each token is routed
    to `top_k` experts, so the assignments a router hands out per tick are
    `tokens * top_k`, shared between `n_experts`.
    """
    return tokens_per_tick * top_k / n_experts


def resolve_threshold(threshold: int | None, threshold_rel: float | None,
                      tokens_per_tick: int, top_k: int, n_experts: int) -> int | None:
    """Q in tokens, from either the absolute or the relative setting.

    The relative form is the useful one: at tiny scale a balanced expert already
    sees ~3k tokens per micro-batch and ~49k per step, so an absolute threshold
    copied from a paper written at another scale either fires every tick or never.
    `threshold_rel=k` asks for windows of about k ticks for an average expert.
    """
    if (threshold is None) == (threshold_rel is None):
        raise ValueError("set exactly one of threshold (tokens) and threshold_rel "
                         "(multiples of a balanced expert's per-tick load), got "
                         f"threshold={threshold}, threshold_rel={threshold_rel}")
    if threshold is not None:
        if threshold < 1:
            raise ValueError(f"threshold must be at least 1 token, got {threshold}")
        return int(threshold)
    if threshold_rel <= 0:
        raise ValueError(f"threshold_rel must be positive, got {threshold_rel}")
    return max(1, round(threshold_rel * mean_tokens_per_expert(tokens_per_tick, top_k,
                                                               n_experts)))


# ──────────────────────────────────────────────────────────────────────
# The controller
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Event:
    """Which experts close their window this tick, and why."""
    fire: torch.Tensor          # [L, E] bool
    by_threshold: torch.Tensor  # [L, E] bool -- reached Q
    by_timeout: torch.Tensor    # [L, E] bool -- hit D with something to show for it

    @property
    def any(self) -> bool:
        return bool(self.fire.any())


class AccumState:
    """Per-expert window bookkeeping: `[L, E]` counters, nothing else.

    `pending` is the token count of the open window and `age` the number of ticks
    it has been open. Both reset on an event. `events` and `last_event` are for the
    dashboard and the JSONL; they never feed a decision.

    The *gradient* itself is not here: it lives in `p.grad`, which the trainer
    stops zeroing in accumulation mode (see moestack/callbacks.py). Keeping the
    accumulator in the gradient buffer is what makes this cost no extra memory.
    """

    def __init__(self, pending, age, events, last_event):
        self.pending, self.age = pending, age
        self.events, self.last_event = events, last_event

    @classmethod
    def zeros(cls, n_layers: int, n_experts: int, device=None) -> "AccumState":
        z = lambda fill: torch.full((n_layers, n_experts), fill, dtype=torch.long,  # noqa: E731
                                    device=device)
        return cls(z(0), z(0), z(0), z(-1))

    @property
    def shape(self):
        return tuple(self.pending.shape)

    def to(self, device) -> "AccumState":
        self.pending, self.age = self.pending.to(device), self.age.to(device)
        self.events, self.last_event = self.events.to(device), self.last_event.to(device)
        return self

    # -- the three operations ------------------------------------------
    def observe(self, counts: torch.Tensor) -> None:
        """Add one tick's routed-token counts to every open window."""
        if tuple(counts.shape) != self.shape:
            raise ValueError(f"counts {tuple(counts.shape)} must match the state's "
                             f"{self.shape}")
        self.pending = self.pending + counts.to(self.pending)
        # age is ticks since the window's *first token*: an expert sitting at zero
        # is not waiting, it simply has no window open yet.
        self.age = torch.where(self.pending > 0, self.age + 1, torch.zeros_like(self.age))

    def ready(self, threshold: int | None, max_window: int | None) -> Event:
        if threshold is None and max_window is None:
            raise ValueError("an accumulation rule needs a threshold, a max_window, "
                             "or both -- with neither, no expert would ever update")
        open_ = self.pending > 0
        by_threshold = (open_ & (self.pending >= threshold) if threshold is not None
                        else torch.zeros_like(open_))
        # `open_` is what stops a dead expert from being fired by the timeout. It is
        # implied by age > 0 today, but it is the rule, so it is written as the rule.
        by_timeout = (open_ & (self.age >= max_window) if max_window is not None
                      else torch.zeros_like(open_))
        return Event(fire=by_threshold | by_timeout,
                     by_threshold=by_threshold,
                     by_timeout=by_timeout & ~by_threshold)

    def consume(self, fire: torch.Tensor, step: int) -> torch.Tensor:
        """Close the firing windows. Returns the tokens each event was built from
        (`[L, E]`, zero where nothing fired) -- the caller needs those *before* the
        reset, both to scale the gradient and to log the realised budget."""
        if tuple(fire.shape) != self.shape:
            raise ValueError(f"fire {tuple(fire.shape)} must match the state's {self.shape}")
        tokens = torch.where(fire, self.pending, torch.zeros_like(self.pending))
        zero = torch.zeros_like(self.pending)
        self.pending = torch.where(fire, zero, self.pending)
        self.age = torch.where(fire, zero, self.age)
        self.events = self.events + fire.to(self.events)
        self.last_event = torch.where(fire, torch.full_like(self.last_event, step),
                                      self.last_event)
        return tokens

    def flush(self) -> Event:
        """Fire every open window regardless of Q and D: the end of training, where
        a window that never filled would otherwise be thrown away. Still subject to
        the one rule that is never relaxed -- an expert with no tokens does not step.
        """
        open_ = self.pending > 0
        return Event(fire=open_, by_threshold=torch.zeros_like(open_), by_timeout=open_)

    # -- checkpointing -------------------------------------------------
    def state_dict(self) -> dict:
        return {"pending": self.pending.cpu(), "age": self.age.cpu(),
                "events": self.events.cpu(), "last_event": self.last_event.cpu()}

    def load_state_dict(self, sd: dict) -> None:
        device = self.pending.device
        self.pending, self.age = sd["pending"].to(device), sd["age"].to(device)
        self.events, self.last_event = sd["events"].to(device), sd["last_event"].to(device)


# ──────────────────────────────────────────────────────────────────────
# Turning an event into numbers the optimizer and the dashboard want
# ──────────────────────────────────────────────────────────────────────

def grad_scale(tokens: torch.Tensor, fire: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """What to multiply each expert's accumulated gradient by, `[L, E]` float.

    Exactly 1.0 on every expert that is *not* firing, because the caller applies
    this to whole `[E, out, in]` tensors: scaling a slice that is still filling its
    window would corrupt it. `mode="sum"` is therefore all ones, and the caller can
    skip the multiply entirely.
    """
    if mode not in GRAD_SCALES:
        raise ValueError(f"grad_scale must be one of {GRAD_SCALES}, got {mode!r}")
    ones = torch.ones(tokens.shape, dtype=torch.float32, device=tokens.device)
    if mode == "sum":
        return ones
    safe = tokens.to(torch.float32).clamp_min(1.0)      # fire => tokens >= 1 anyway
    return torch.where(fire, 1.0 / safe, ones)


def accum_metrics(state: AccumState, event: Event, tokens: torch.Tensor,
                  threshold: int | None) -> dict:
    """Aggregates for wandb, read *after* `consume`: `tokens` is what the events
    were built from, while `state` is what is still waiting.

    Per-expert series are far too wide to log (64 experts x 16 layers), so the
    dashboard gets these and the joint distribution goes to the JSONL -- the same
    division of labour as `expert_lr.scale_summary`.
    """
    nan = float("nan")
    fire, n_fired = event.fire, int(event.fire.sum())
    fired_tokens = tokens[fire].to(torch.float32) if n_fired else tokens.new_zeros(0).float()
    pending, age = state.pending.float(), state.age.float()
    out = {
        # Updates per expert per tick: 1.0 is the baseline's "everyone, every step".
        "accum/event_fraction": float(fire.to(torch.float32).mean()),
        "accum/waiting_fraction": float((state.pending > 0).to(torch.float32).mean()),
        "accum/never_fired_fraction": float((state.events == 0).to(torch.float32).mean()),
        "accum/pending_mean": float(pending.mean()),
        "accum/pending_max": float(pending.max()) if pending.numel() else nan,
        # The staleness bound in ticks. If this sits at max_window, the timeout --
        # not the token budget -- is what is driving the experiment.
        "accum/age_max": float(age.max()) if age.numel() else nan,
    }
    if n_fired:
        out.update({
            "accum/tokens_per_event_mean": float(fired_tokens.mean()),
            "accum/tokens_per_event_min": float(fired_tokens.min()),
            "accum/tokens_per_event_max": float(fired_tokens.max()),
            # Of the events that happened, how many were the timeout giving up
            # rather than the budget being met. A high number means Q is out of
            # range for this run and the windows are really fixed-length.
            "accum/timeout_fraction": float(event.by_timeout.to(torch.float32).sum() / n_fired),
        })
        if threshold:
            # How far past Q a window lands, because a single tick can carry an
            # expert well over the line. Far above 1.0 means the tick is too coarse
            # for the budget being asked for.
            out["accum/overshoot_mean"] = float(fired_tokens.mean() / threshold)
    return out
