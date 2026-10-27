"""Data sources. Each yields (input_ids, targets) micro-batches of shape [B, T].

    stream     the repo-root `data.py` pipeline (DCLM-edu / FinePhrase mix, cl100k),
               unchanged. Sharded over all ranks, which equals DP ranks here. When
               EP/TP arrive, sharding must follow the *DP* rank only: ranks inside
               one DP group must see the same tokens.
    disk       a flat uint32 token file built once by `moestack.prepare_data` from
               the same stream. Seekable, no network, no DataLoader workers -- the
               tokens are memory-mapped and indexed arithmetically. Preferred for
               experiments: identical sequences to `stream`, without the ~50 min of
               shuffle-buffer filling or the risk of a worker hanging mid-run.
    synthetic  deterministic random tokens. Seekable, needs no network or tokenizer;
               for smoke tests, CI and throughput measurements.

A source is wrapped in `MicroBatchIterator`, which counts consumed micro-batches so
the position can be checkpointed. The stream is not seekable, so resuming replays
it (as the original trainer did); synthetic resumes by skipping instantly.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .config import DataConfig

# `data.get_eval_batches` reaches its batches by skipping this many tokens into the
# stream. The disk source used to reproduce that offset exactly, so that disk and
# stream runs evaluated on the same text -- see `DiskSource.eval_batches` for why
# it no longer does. Kept because the stream path still goes through upstream.
EVAL_SKIP_TOKENS = 10_000_000


class SyntheticSource:
    """Random tokens with a Zipf-ish unigram distribution so the loss can drop.
    Micro-batch i on rank r is a pure function of (seed, r, i)."""

    def __init__(self, vocab_size: int, cfg: DataConfig, seed: int, rank: int):
        self.vocab, self.cfg, self.seed, self.rank = vocab_size, cfg, seed, rank
        ranks = torch.arange(1, vocab_size + 1, dtype=torch.float)
        self.probs = (1.0 / ranks) / (1.0 / ranks).sum()

    def batch(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.rank * 7919 + i)
        B, T = self.cfg.batch_size, self.cfg.seq_len
        toks = torch.multinomial(self.probs, B * (T + 1), replacement=True, generator=g).view(B, T + 1)
        return toks[:, :-1].contiguous(), toks[:, 1:].contiguous()

    def iterator(self, start: int = 0) -> Iterator:
        i = start
        while True:
            yield self.batch(i)
            i += 1


class DiskSource:
    """Memory-mapped flat token file written by `moestack.prepare_data`.

    Sequences are carved exactly as the streaming pipeline carves them: contiguous,
    non-overlapping windows of seq_len+1 tokens yielding (w[:-1], w[1:]). Global
    micro-batch order matches a DDP stream run -- at optimizer step s, micro-step j,
    rank r reads global micro-batch (s*accum + j)*world + r -- so ranks never
    overlap and the global batch is the same set of sequences at any world size.
    """

    def __init__(self, cache_dir: str, cfg: DataConfig, rank: int = 0, world_size: int = 1):
        d = Path(cache_dir)
        meta_path, tok_path = d / "meta.json", d / "tokens.u32"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no token cache at {d}. Build one with:\n"
                f"  python -m moestack.prepare_data --out {d}")
        self.meta = json.loads(meta_path.read_text())
        if self.meta.get("format") != "moestack-tokens-v1":
            raise ValueError(f"{meta_path} is not a moestack token cache")
        for key, want in (("tokenizer", cfg.tokenizer),
                          ("insert_separator", cfg.insert_separator)):
            if self.meta[key] != want:
                raise ValueError(f"cache was built with {key}={self.meta[key]!r} but this "
                                 f"run asks for {want!r}; rebuild it or change the config")
        self.cfg, self.rank, self.world_size = cfg, rank, world_size
        self.stride = cfg.seq_len + 1
        self.tokens = np.memmap(tok_path, dtype=np.uint32, mode="r",
                                shape=(self.meta["n_tokens"],))
        self.n_seq = self.meta["n_tokens"] // self.stride
        if self.n_seq < cfg.batch_size:
            raise ValueError(f"cache holds {self.n_seq} sequences at seq_len={cfg.seq_len}; "
                             f"need at least one batch of {cfg.batch_size}")

    def _sequences(self, first: int, count: int):
        """(inp, tgt) for `count` consecutive sequences starting at index `first`,
        wrapping if the cache runs out (as restarting the stream would)."""
        idx = (first + np.arange(count)) % self.n_seq
        rows = self.tokens[(idx[:, None] * self.stride) + np.arange(self.stride)]
        rows = torch.from_numpy(rows.astype(np.int64))
        return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()

    def batch(self, i: int):
        """Micro-batch `i` as counted by *this* rank."""
        g = i * self.world_size + self.rank
        return self._sequences(g * self.cfg.batch_size, self.cfg.batch_size)

    def iterator(self, start: int = 0) -> Iterator:
        i = start
        while True:
            yield self.batch(i)
            i += 1

    def eval_batches(self, n_batches: int, train_sequences: int = 0) -> list:
        """The **tail** of the cache: the sequences a run never reaches.

        This used to be the slice `data.get_eval_batches` returns -- `EVAL_SKIP_TOKENS`
        into the stream -- so that disk and stream runs scored the same text. That
        offset is sequence 4,882 at seq_len=2048, and training reads 256 sequences a
        step at 16x16, so **training overran the entire eval set by step 20** and
        every `eval/loss` after it was measuring memorisation. A 10M-token skip is a
        20-step head start, not a holdout.

        The tail is held out by construction for any run that does not wrap the
        cache, and `train_sequences` makes that condition explicit rather than
        assumed: pass the number of sequences the whole run will read and this
        refuses to hand back a slice the run would train on. The training stream is
        untouched -- only the eval set moves -- so `train/lm_loss` still lines up
        with runs logged before the change, while their `eval/loss` does not.
        """
        need = n_batches * self.cfg.batch_size
        if need > self.n_seq:
            raise ValueError(
                f"cache holds {self.n_seq:,} sequences, fewer than the {need:,} the "
                f"eval slice needs. Rebuild with more --tokens.")
        first = self.n_seq - need
        if train_sequences > first:
            raise ValueError(
                f"the run reads {train_sequences:,} sequences but the held-out tail "
                f"starts at {first:,}, so it would train on its own eval set. Shorten "
                f"the run, lower data.eval_batches, or rebuild the cache with more "
                f"--tokens.")
        inp, tgt = self._sequences(first, need)
        return list(zip(inp.split(self.cfg.batch_size), tgt.split(self.cfg.batch_size)))


class MicroBatchIterator:
    def __init__(self, make_iter, seekable: bool, start: int = 0):
        self._make_iter, self.seekable = make_iter, seekable
        self.consumed = 0
        self._it = make_iter(start) if seekable else make_iter(0)
        if start and not seekable:
            for _ in range(start):
                self._next_raw()
        self.consumed = start

    def _next_raw(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = self._make_iter(0)
            return next(self._it)

    def __next__(self):
        b = self._next_raw()
        self.consumed += 1
        return b

    def __iter__(self):
        return self

    def state_dict(self):
        return {"consumed": self.consumed}


def build_train_iterator(cfg: DataConfig, vocab_size: int, seed: int, rank: int,
                         start: int = 0, world_size: int = 1) -> MicroBatchIterator:
    if cfg.source == "disk":
        src = DiskSource(cfg.cache_dir, cfg, rank, world_size)
        return MicroBatchIterator(src.iterator, seekable=True, start=start)
    if cfg.source == "synthetic":
        src = SyntheticSource(vocab_size, cfg, seed, rank)
        return MicroBatchIterator(src.iterator, seekable=True, start=start)
    if cfg.source == "stream":
        from data import get_dataloader     # repo-root data.py
        loader = get_dataloader(seq_len=cfg.seq_len, batch_size=cfg.batch_size,
                                num_workers=cfg.num_workers, tokenizer=cfg.tokenizer,
                                insert_separator=cfg.insert_separator)
        return MicroBatchIterator(lambda _start: iter(loader), seekable=False, start=start)
    raise ValueError(f"unknown data source {cfg.source!r}")


def build_eval_batches(cfg: DataConfig, vocab_size: int, seed: int,
                       train_sequences: int = 0) -> list:
    """`train_sequences` is how many sequences the whole run reads; the disk source
    uses it to prove its slice is one the run never trains on."""
    if cfg.source == "disk":
        # rank 0 / world 1: every rank evaluates on the identical slice.
        return DiskSource(cfg.cache_dir, cfg).eval_batches(cfg.eval_batches,
                                                           train_sequences)
    if cfg.source == "synthetic":
        # Rank -1 keeps eval disjoint from every training stream and identical on all ranks.
        src = SyntheticSource(vocab_size, cfg, seed, rank=-1)
        return [src.batch(i) for i in range(cfg.eval_batches)]
    # Upstream skips EVAL_SKIP_TOKENS into the same stream training reads from, and
    # there is no tail to retreat to, so this slice is held out only for a run short
    # enough never to reach it. Warned, not raised: the stream path predates this and
    # is not what the configs use.
    if train_sequences * cfg.seq_len > EVAL_SKIP_TOKENS:
        warnings.warn(
            f"data.source=stream evaluates {EVAL_SKIP_TOKENS:,} tokens into the "
            f"training stream, which this run of ~{train_sequences * cfg.seq_len:,} "
            f"tokens reads past: eval/loss will score text the model has trained on. "
            f"Use data.source=disk for a held-out tail.", RuntimeWarning, stacklevel=2)
    from data import get_eval_batches
    return get_eval_batches(seq_len=cfg.seq_len, batch_size=cfg.batch_size,
                            n_batches=cfg.eval_batches, tokenizer=cfg.tokenizer,
                            insert_separator=cfg.insert_separator)
