"""Data sources. Each yields (input_ids, targets) micro-batches of shape [B, T].

    stream     the repo-root `data.py` pipeline (DCLM-edu / FinePhrase mix, cl100k),
               unchanged. Sharded over all ranks, which equals DP ranks here. When
               EP/TP arrive, sharding must follow the *DP* rank only: ranks inside
               one DP group must see the same tokens.
    synthetic  deterministic random tokens. Seekable, needs no network or tokenizer;
               for smoke tests, CI and throughput measurements.

A source is wrapped in `MicroBatchIterator`, which counts consumed micro-batches so
the position can be checkpointed. The stream is not seekable, so resuming replays
it (as the original trainer did); synthetic resumes by skipping instantly.
"""

from __future__ import annotations

from typing import Iterator

import torch

from .config import DataConfig


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
                         start: int = 0) -> MicroBatchIterator:
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


def build_eval_batches(cfg: DataConfig, vocab_size: int, seed: int) -> list:
    if cfg.source == "synthetic":
        # Rank -1 keeps eval disjoint from every training stream and identical on all ranks.
        src = SyntheticSource(vocab_size, cfg, seed, rank=-1)
        return [src.batch(i) for i in range(cfg.eval_batches)]
    from data import get_eval_batches
    return get_eval_batches(seq_len=cfg.seq_len, batch_size=cfg.batch_size,
                            n_batches=cfg.eval_batches, tokenizer=cfg.tokenizer,
                            insert_separator=cfg.insert_separator)
