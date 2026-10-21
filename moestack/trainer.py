"""The training loop. Deliberately boring: no MoE-specific logic, no experiment
logic. If an idea needs to change something here, that usually means a missing
callback event -- add the event rather than special-casing the loop."""

from __future__ import annotations

import json
import math
import os
import signal
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from . import checkpoint
from .callbacks import (CallbackList, ConsoleLogger, ExpertStats, MoELoadStats, PerExpertLR,
                        StepState, WandbLogger)
from .config import Config
from .data import build_eval_batches, build_train_iterator
from .initialization import init_weights
from .losses import compute_loss, lm_cross_entropy_sum
from .model import Transformer
from .optim import build_optimizers, lr_at
from .parallel import ParallelDims, all_reduce_, init_distributed, parallelize


class Trainer:
    def __init__(self, cfg: Config, callbacks: list | None = None, verbose: bool = True):
        self.cfg = cfg
        self.dist = d = init_distributed(cfg.train.device)
        self.verbose = verbose and d.is_main
        if cfg.train.dry_run:
            cfg.train.n_steps, cfg.train.eval_every = 100, 50
        # Computed after dry_run is applied, because n_steps feeds the fingerprint.
        # train.py takes the name from here rather than recomputing it.
        self.run_name = cfg.run_name()
        self.save_dir = Path(cfg.train.save_dir)

        self.dims = ParallelDims.from_config(cfg.parallel, d.world_size)
        self.mesh = self.dims.build_mesh(d.device.type)

        # --- model: identical init on every rank (DDP also broadcasts rank 0's) ---
        torch.manual_seed(cfg.train.seed)
        self.args = cfg.model.build_args()
        self.model = Transformer(self.args)
        init_weights(self.model)
        self.model.to(d.device)
        self.fwd = parallelize(self.model, cfg.parallel, d)    # what we call forward on
        self.n_params = sum(p.numel() for p in self.model.parameters())

        # split_experts puts each layer's expert tensors in their own param groups,
        # which is what makes them individually addressable; it changes no arithmetic.
        self.optimizers = build_optimizers(self.model, cfg.optim, verbose=self.verbose,
                                           split_experts=cfg.optim.per_expert_lr.enabled)

        cbs = [MoELoadStats(cfg.logging.load_stats_per_layer)] if cfg.logging.load_stats else []
        if cfg.optim.per_expert_lr.enabled:
            cbs.append(PerExpertLR(cfg.optim.per_expert_lr))
        if cfg.logging.expert_stats_every:
            cbs.append(ExpertStats(cfg.logging.expert_stats_every))
        cbs += list(callbacks or [])
        cbs.append(ConsoleLogger(cfg.train.log_every))
        if cfg.logging.wandb and d.is_main:
            cbs.append(WandbLogger(cfg.logging, self.run_name, cfg.to_dict(),
                                   group=cfg.condition_name()))
        self.callbacks = CallbackList(cbs)

        self.use_amp = cfg.parallel.mixed_precision == "bf16" and d.device.type == "cuda"
        self.tokens_per_step = (cfg.data.batch_size * cfg.data.seq_len *
                                cfg.data.grad_accum * self.dims.dp)

        # --- state, possibly restored ---
        self.step, self.tokens_seen, self.best_eval = 0, 0, float("inf")
        data_start, st_skipped = 0, 0
        resume = cfg.train.resume
        if cfg.train.auto_resume and not resume:
            found = checkpoint.latest(self.save_dir, self.run_name)
            resume = str(found) if found else None
        if resume:
            st = checkpoint.load(Path(resume), model=self.model, optimizers=self.optimizers,
                                 callbacks=self.callbacks)
            self.step, self.tokens_seen = st["step"], st["tokens_seen"]
            self.best_eval, data_start = st["best_eval"], st["data"]["consumed"]
            st_skipped = st.get("skipped", 0)
            if self.verbose:
                print(f"Resumed from {resume} at step {self.step}")
        # Non-finite steps are dropped, not applied (see train_step).
        self.skipped, self.consecutive_skips = st_skipped, 0
        self._stop_requested = False
        # Parameters whose `.grad` is a multi-batch accumulator rather than this
        # batch's gradient, registered by a callback at on_train_start. They are
        # spared the per-step clear and left out of the gradient-norm clip, because
        # both would corrupt a window that is still filling. Empty by default, and
        # the trainer never asks what the accumulation rule is -- see
        # moestack/expert_accum.py.
        self.grad_accumulators: list[nn.Parameter] = []

        self.train_iter = build_train_iterator(cfg.data, self.args.vocab_size, cfg.train.seed,
                                               d.rank, start=data_start,
                                               world_size=self.dims.dp)
        self.eval_batches = None    # fetched lazily: the stream burns 10M tokens first

        if self.verbose:
            print(f"run={self.run_name} params={self.n_params:,} world={d.world_size} "
                  f"device={d.device} amp={self.use_amp} tokens/step={self.tokens_per_step:,}")
            # The fingerprint in the run name stands for exactly this list.
            over = cfg.significant_overrides()
            print("  config: " + (" ".join(f"{k}={v}" for k, v in over.items()) or "(defaults)"))

    # ------------------------------------------------------------------
    def _amp(self):
        return (torch.autocast("cuda", dtype=torch.bfloat16) if self.use_amp else nullcontext())

    def train_step(self) -> StepState:
        cfg, d = self.cfg, self.dist
        accum = cfg.data.grad_accum
        self.optimizers.zero_grad(set_to_none=True, keep=self.grad_accumulators)
        sums: dict[str, torch.Tensor] = {}
        counts = torch.zeros(self.args.n_layers, self.args.n_experts, dtype=torch.long,
                             device=d.device)

        # Before the micro-batch loop, not after it: a callback that steps part of
        # the model mid-batch has to see this step's scheduled rate. The value does
        # not depend on anything the loop computes, so moving it changes nothing.
        lr = lr_at(self.step, cfg.optim, cfg.train.n_steps)
        self.optimizers.set_lr(lr)
        state = StepState(step=self.step, lr=lr)

        for i in range(accum):
            inp, tgt = next(self.train_iter)
            inp, tgt = inp.to(d.device, non_blocking=True), tgt.to(d.device, non_blocking=True)
            sync = (self.fwd.no_sync() if d.enabled and i < accum - 1 else nullcontext())
            with sync:
                with self._amp():
                    out = self.fwd(inp)
                    loss, terms = compute_loss(out, tgt, cfg.loss)
                (loss / accum).backward()
            for t in terms:
                sums[t.name] = sums.get(t.name, 0.0) + t.value.detach().float() / accum
            sums["total"] = sums.get("total", 0.0) + loss.detach().float() / accum
            micro_counts = torch.stack([r.counts for r in out.router_outs])
            counts += micro_counts
            del out, terms, loss, inp, tgt  # free the memory before the all_reduce
            # Fired after the activations are freed, so a callback that steps part
            # of the model here is not doing it at the peak of the micro-batch.
            self.callbacks.on_microbatch_end(self, state, i, micro_counts)

        # One collective for all scalars, one for the counts. Counts are reduced
        # BEFORE anything is derived from them, so every rank sees the same numbers.
        names = list(sums)
        vec = all_reduce_(torch.stack([sums[n] for n in names]))
        vec = vec / d.world_size
        all_reduce_(counts)

        grad_norm = nn.utils.clip_grad_norm_(self._clipped_params(), cfg.optim.grad_clip)

        values = dict(zip(names, vec.tolist()))
        total = values.pop("total")
        self.tokens_seen += self.tokens_per_step
        state.loss_terms, state.grad_norm = values, float(grad_norm)
        state.expert_counts, state.tokens_seen = counts, self.tokens_seen
        state.metrics["train/loss"] = total
        self.callbacks.on_before_optimizer_step(self, state)

        # Applying a non-finite step poisons the weights, and the next `save()`
        # would then write the poisoned weights to disk. Drop it instead; grads are
        # zeroed at the top of the next step, so nothing leaks into it.
        if not (math.isfinite(state.grad_norm) and math.isfinite(total)):
            self.skipped += 1
            self.consecutive_skips += 1
            state.metrics["train/skipped_steps"] = self.skipped
            if self.verbose:
                print(f"[step {self.step}] non-finite loss={total} gnorm={state.grad_norm} "
                      f"-- step skipped ({self.consecutive_skips} in a row)", flush=True)
            if self.consecutive_skips >= cfg.train.max_consecutive_skips:
                raise FloatingPointError(
                    f"{self.consecutive_skips} consecutive non-finite steps ending at "
                    f"step {self.step}; loss={total} grad_norm={state.grad_norm}")
            return state

        self.consecutive_skips = 0
        state.metrics["train/skipped_steps"] = self.skipped
        # Past the drop: anything that moves weights or consumes accumulated state
        # happens from here, never from on_before_optimizer_step.
        self.callbacks.on_accepted_step(self, state)
        self.optimizers.step()
        return state

    def _clipped_params(self):
        """Everything clip_grad_norm_ should see. An accumulator's `.grad` spans
        several batches, so scaling it would not clip a gradient -- it would shrink
        a window that is still filling, and differently each time it is visited."""
        if not self.grad_accumulators:
            return self.model.parameters()
        skip = {id(p) for p in self.grad_accumulators}
        return [p for p in self.model.parameters() if id(p) not in skip]

    @torch.no_grad()
    def evaluate(self) -> dict:
        if self.eval_batches is None:
            self.eval_batches = build_eval_batches(self.cfg.data, self.args.vocab_size,
                                                   self.cfg.train.seed)
        self.model.eval()
        # Same scoring rule as training, via the same function: -100 is ignored and
        # excluded from the denominator, and the chunking caps the fp32 transient.
        tot, n = 0.0, 0
        for inp, tgt in self.eval_batches:
            inp, tgt = inp.to(self.dist.device), tgt.to(self.dist.device)
            with self._amp():
                logits = self.model(inp).logits
            batch_sum, batch_n = lm_cross_entropy_sum(logits, tgt, self.cfg.loss.lm_chunk_rows)
            tot += batch_sum.item()
            n += int(batch_n)
        self.model.train()
        loss = tot / max(n, 1)
        return {"loss": loss, "perplexity": math.exp(min(loss, 20))}

    def save(self, name: str):
        if not self.dist.is_main:
            return
        path = self.save_dir / f"{self.run_name}_{name}.pt"
        checkpoint.save(path, model=self.model, optimizers=self.optimizers,
                        trainer_state={"step": self.step, "tokens_seen": self.tokens_seen,
                                       "best_eval": self.best_eval,
                                       "skipped": self.skipped,
                                       "data": self.train_iter.state_dict()},
                        callbacks=self.callbacks, config=self.cfg.to_dict())
        if self.verbose:
            print(f"  -> {path}")

    # ------------------------------------------------------------------
    @contextmanager
    def _stop_on_signals(self):
        """SIGTERM/SIGINT ask the loop to checkpoint and exit at the next step
        boundary instead of dying where they land.

        The cluster wrapper caps every job with `timeout`, which sends SIGTERM and
        lets runai retry; without this each expiry throws away up to `save_every`
        steps. A second signal restores the default handler and kills immediately,
        so a hung step is still interruptible.
        """
        self._stop_requested = False
        previous = {}

        def handler(signum, _frame):
            if self._stop_requested:                   # second signal: don't wait
                signal.signal(signum, previous.get(signum, signal.SIG_DFL))
                os.kill(os.getpid(), signum)
            self._stop_requested = True

        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    previous[sig] = signal.signal(sig, handler)
                except ValueError:
                    pass            # not the main thread: no handler, no graceful stop
            yield
        finally:
            for sig, prev in previous.items():
                signal.signal(sig, prev)

    def _stop_agreed(self) -> bool:
        """Every rank has to reach the same verdict: if one breaks out of the loop
        and the others do not, the others hang on the next collective."""
        if not self.dist.enabled:
            return self._stop_requested
        flag = torch.tensor([1.0 if self._stop_requested else 0.0], device=self.dist.device)
        all_reduce_(flag)
        return bool(flag.item() > 0)

    def fit(self) -> dict:
        cfg = self.cfg.train
        self.model.train()
        self.callbacks.on_train_start(self)
        t0, state = time.time(), StepState(step=self.step)
        m, stopped = None, False
        with self._stop_on_signals():
            while self.step < cfg.n_steps:
                state = self.train_step()
                self.callbacks.on_step_end(self, state)
                self.step += 1      # = completed steps = index of the next step to run
                m = None
                if cfg.eval_every and self.step % cfg.eval_every == 0:
                    m = self.evaluate()
                    self.callbacks.on_eval(self, state, m)
                    if m["loss"] < self.best_eval:
                        self.best_eval = m["loss"]
                        self.save("best")
                just_saved = bool(cfg.save_every and self.step % cfg.save_every == 0)
                if just_saved:
                    self.save(f"step{self.step}")
                if self._stop_agreed():
                    stopped = True
                    if self.verbose:
                        print(f"[step {self.step}] stop signal -- checkpointing and "
                              f"exiting; rerun with train.auto_resume=true", flush=True)
                    if not just_saved:
                        self.save(f"step{self.step}")
                    break

        if stopped:
            # No closing eval: the point is to be gone before the kill lands.
            final = m or {"loss": float("nan"), "perplexity": float("nan")}
        else:
            # Last chance to close any window still open: after this the weights are
            # saved and the run is over. A stopped run deliberately does *not* flush
            # -- its windows are in the checkpoint, and the resumed run continues
            # them. If the flush moved anything, the eval the last step ran no
            # longer describes the model, so it is redone.
            if self.callbacks.on_before_final_eval(self, state):
                m = None
            final = m or self.evaluate()    # don't re-evaluate if the last step just did
            if m is None:
                self.callbacks.on_eval(self, state, final)
            if not (cfg.save_every and self.step % cfg.save_every == 0):
                self.save(f"step{self.step}")
        self.callbacks.on_train_end(self, state)
        results = {"run_name": self.run_name, "final_eval_loss": final["loss"],
                   "best_eval_loss": min(self.best_eval, final["loss"]),
                   "params": self.n_params, "train_seconds": time.time() - t0,
                   "tokens_seen": self.tokens_seen, "skipped_steps": self.skipped,
                   "completed_steps": self.step, "stopped_early": stopped}
        if self.dist.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            (self.save_dir / f"{self.run_name}_results.json").write_text(json.dumps(results, indent=2))
        return results

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()
