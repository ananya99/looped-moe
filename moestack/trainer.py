"""The training loop. Deliberately boring: no MoE-specific logic, no experiment
logic. If an idea needs to change something here, that usually means a missing
callback event -- add the event rather than special-casing the loop."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from . import checkpoint
from .callbacks import CallbackList, ConsoleLogger, MoELoadStats, StepState, WandbLogger
from .config import Config
from .data import build_eval_batches, build_train_iterator
from .initialization import init_weights
from .losses import compute_loss
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

        self.optimizers = build_optimizers(self.model, cfg.optim, verbose=self.verbose)

        cbs = [MoELoadStats()] if cfg.logging.load_stats else []
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
        data_start = 0
        resume = cfg.train.resume
        if cfg.train.auto_resume and not resume:
            found = checkpoint.latest(self.save_dir, self.run_name)
            resume = str(found) if found else None
        if resume:
            st = checkpoint.load(Path(resume), model=self.model, optimizers=self.optimizers,
                                 callbacks=self.callbacks)
            self.step, self.tokens_seen = st["step"], st["tokens_seen"]
            self.best_eval, data_start = st["best_eval"], st["data"]["consumed"]
            if self.verbose:
                print(f"Resumed from {resume} at step {self.step}")

        self.train_iter = build_train_iterator(cfg.data, self.args.vocab_size, cfg.train.seed,
                                               d.rank, start=data_start)
        self.eval_batches = None    # fetched lazily: the stream burns 10M tokens first

        if self.verbose:
            print(f"run={self.run_name} params={self.n_params:,} world={d.world_size} "
                  f"device={d.device} amp={self.use_amp} tokens/step={self.tokens_per_step:,}")

    # ------------------------------------------------------------------
    def _amp(self):
        return (torch.autocast("cuda", dtype=torch.bfloat16) if self.use_amp else nullcontext())

    def train_step(self) -> StepState:
        cfg, d = self.cfg, self.dist
        accum = cfg.data.grad_accum
        self.optimizers.zero_grad(set_to_none=True)
        sums: dict[str, torch.Tensor] = {}
        counts = torch.zeros(self.args.n_layers, self.args.n_experts, dtype=torch.long,
                             device=d.device)

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
            counts += torch.stack([r.counts for r in out.router_outs])

        # One collective for all scalars, one for the counts. Counts are reduced
        # BEFORE anything is derived from them, so every rank sees the same numbers.
        names = list(sums)
        vec = all_reduce_(torch.stack([sums[n] for n in names]))
        vec = vec / d.world_size
        all_reduce_(counts)

        lr = lr_at(self.step, cfg.optim, cfg.train.n_steps)
        self.optimizers.set_lr(lr)
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optim.grad_clip)

        values = dict(zip(names, vec.tolist()))
        total = values.pop("total")
        self.tokens_seen += self.tokens_per_step
        state = StepState(step=self.step, lr=lr, loss_terms=values, grad_norm=float(grad_norm),
                          expert_counts=counts, tokens_seen=self.tokens_seen,
                          metrics={"train/loss": total})
        self.callbacks.on_before_optimizer_step(self, state)
        self.optimizers.step()
        return state

    @torch.no_grad()
    def evaluate(self) -> dict:
        if self.eval_batches is None:
            self.eval_batches = build_eval_batches(self.cfg.data, self.args.vocab_size,
                                                   self.cfg.train.seed)
        self.model.eval()
        tot, n = 0.0, 0
        for inp, tgt in self.eval_batches:
            inp, tgt = inp.to(self.dist.device), tgt.to(self.dist.device)
            with self._amp():
                logits = self.model(inp).logits
            tot += nn.functional.cross_entropy(logits.float().flatten(0, 1), tgt.flatten(),
                                               reduction="sum").item()
            n += tgt.numel()
        self.model.train()
        loss = tot / n
        return {"loss": loss, "perplexity": math.exp(min(loss, 20))}

    def save(self, name: str):
        if not self.dist.is_main:
            return
        path = self.save_dir / f"{self.run_name}_{name}.pt"
        checkpoint.save(path, model=self.model, optimizers=self.optimizers,
                        trainer_state={"step": self.step, "tokens_seen": self.tokens_seen,
                                       "best_eval": self.best_eval,
                                       "data": self.train_iter.state_dict()},
                        callbacks=self.callbacks, config=self.cfg.to_dict())
        if self.verbose:
            print(f"  -> {path}")

    # ------------------------------------------------------------------
    def fit(self) -> dict:
        cfg = self.cfg.train
        self.model.train()
        self.callbacks.on_train_start(self)
        t0, state = time.time(), StepState(step=self.step)
        m = None
        while self.step < cfg.n_steps:
            state = self.train_step()
            self.callbacks.on_step_end(self, state)
            if not math.isfinite(state.metrics["train/loss"]):
                raise FloatingPointError(f"non-finite loss at step {self.step}")
            self.step += 1          # = completed steps = index of the next step to run
            m = None
            if cfg.eval_every and self.step % cfg.eval_every == 0:
                m = self.evaluate()
                self.callbacks.on_eval(self, state, m)
                if m["loss"] < self.best_eval:
                    self.best_eval = m["loss"]
                    self.save("best")
            if cfg.save_every and self.step % cfg.save_every == 0:
                self.save(f"step{self.step}")

        final = m or self.evaluate()        # don't re-evaluate if the last step just did
        if m is None:
            self.callbacks.on_eval(self, state, final)
        if not (cfg.save_every and self.step % cfg.save_every == 0):
            self.save(f"step{self.step}")
        self.callbacks.on_train_end(self, state)
        results = {"run_name": self.run_name, "final_eval_loss": final["loss"],
                   "best_eval_loss": min(self.best_eval, final["loss"]),
                   "params": self.n_params, "train_seconds": time.time() - t0,
                   "tokens_seen": self.tokens_seen}
        if self.dist.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            (self.save_dir / f"{self.run_name}_results.json").write_text(json.dumps(results, indent=2))
        return results

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()
