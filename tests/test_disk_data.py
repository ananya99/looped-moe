"""The on-disk token cache: carving, DDP sharding, resume, and the eval slice.

The thing these pin down is that `source=disk` is not a different dataset -- it is
the same sequences a `source=stream` run would have seen, read from a file.
"""

import json
import math

import numpy as np
import pytest
import torch

from moestack.config import Config, DataConfig
from moestack.data import EVAL_SKIP_TOKENS, DiskSource, build_train_iterator
from moestack.trainer import Trainer

from conftest import TINY, tiny_config


def make_cache(tmp_path, n_tokens, seq_len=8, vocab=256, tokenizer="cl100k",
               insert_separator="before"):
    """A cache whose token at position i is simply i % vocab, so any carving
    mistake shows up as an arithmetic mismatch rather than a plausible tensor."""
    d = tmp_path / "cache"
    d.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(d / "tokens.u32", dtype=np.uint32, mode="w+", shape=(n_tokens,))
    arr[:] = np.arange(n_tokens, dtype=np.uint32) % vocab
    arr.flush()
    del arr
    (d / "meta.json").write_text(json.dumps({
        "format": "moestack-tokens-v1", "dtype": "uint32", "seq_len": seq_len,
        "tokenizer": tokenizer, "insert_separator": insert_separator,
        "separator_id": 100257, "vocab_size": vocab, "n_tokens": n_tokens}))
    return d


def cfg_for(cache, seq_len=8, batch_size=2, **kw):
    c = DataConfig(source="disk", cache_dir=str(cache), seq_len=seq_len,
                   batch_size=batch_size, **kw)
    return c


# ── carving ──────────────────────────────────────────────────────────

def test_sequences_are_contiguous_non_overlapping_windows(tmp_path):
    """`data.py` buffers seq_len+1 tokens, yields (w[:-1], w[1:]), then clears the
    buffer -- so windows are adjacent and stride by seq_len+1, not seq_len."""
    T, B = 8, 2
    cache = make_cache(tmp_path, n_tokens=(T + 1) * 20, seq_len=T, vocab=10_000)
    src = DiskSource(str(cache), cfg_for(cache, T, B))
    inp, tgt = src.batch(0)
    assert inp.shape == (B, T) and tgt.shape == (B, T)
    # targets are inputs shifted by one
    assert torch.equal(inp[:, 1:], tgt[:, :-1])
    # window k starts at token k*(T+1), not k*T
    for b in range(B):
        assert int(inp[b, 0]) == b * (T + 1)
    # consecutive micro-batches continue where the previous one stopped
    nxt, _ = src.batch(1)
    assert int(nxt[0, 0]) == B * (T + 1)


def test_prepare_data_window_reconstruction_is_exact():
    """prepare_data rebuilds the flat stream as cat(inp, tgt[-1:]). That is only
    correct because windows do not overlap; this is the assumption it rests on."""
    T = 8
    stream = torch.arange(3 * (T + 1))
    windows = stream.view(3, T + 1)
    inp, tgt = windows[:, :-1], windows[:, 1:]
    rebuilt = torch.cat([inp, tgt[:, -1:]], dim=1).reshape(-1)
    assert torch.equal(rebuilt, stream)


# ── DDP sharding ─────────────────────────────────────────────────────

def test_ranks_see_disjoint_sequences_covering_the_same_global_batch(tmp_path):
    T, B, accum = 8, 2, 3
    cache = make_cache(tmp_path, n_tokens=(T + 1) * 400, seq_len=T, vocab=10_000)
    c = cfg_for(cache, T, B)
    one = DiskSource(str(cache), c, rank=0, world_size=1)
    r0 = DiskSource(str(cache), c, rank=0, world_size=2)
    r1 = DiskSource(str(cache), c, rank=1, world_size=2)

    def firsts(src, n):
        return [int(src.batch(i)[0][b, 0]) for i in range(n) for b in range(B)]

    a, b = firsts(r0, accum), firsts(r1, accum)
    assert not set(a) & set(b), "ranks must not share sequences"
    # 2 ranks x accum micro-batches == 1 rank x 2*accum micro-batches, same set
    assert sorted(a + b) == sorted(firsts(one, 2 * accum))


# ── resume ───────────────────────────────────────────────────────────

def test_seeking_equals_consuming(tmp_path):
    T, B = 8, 2
    cache = make_cache(tmp_path, n_tokens=(T + 1) * 200, seq_len=T, vocab=10_000)
    c = cfg_for(cache, T, B)
    straight = build_train_iterator(c, 10_000, seed=0, rank=0, start=0, world_size=1)
    for _ in range(5):
        next(straight)
    seeked = build_train_iterator(c, 10_000, seed=0, rank=0, start=5, world_size=1)
    assert seeked.consumed == 5
    for _ in range(3):
        x, y = next(straight)
        a, b = next(seeked)
        assert torch.equal(x, a) and torch.equal(y, b)


# ── eval slice ───────────────────────────────────────────────────────

def test_eval_offset_matches_upstream_get_eval_batches():
    """Replicate data.get_eval_batches' skip loop and check DiskSource lands on the
    same first sequence. It counts seq_len (not seq_len+1) per sequence."""
    for seq_len in (512, 1024, 2048):
        skipped = 0
        count = 0
        while True:
            count += seq_len
            if count < EVAL_SKIP_TOKENS:
                skipped += 1
            else:
                break
        assert math.ceil(EVAL_SKIP_TOKENS / seq_len) - 1 == skipped, seq_len
    assert math.ceil(EVAL_SKIP_TOKENS / 2048) - 1 == 4882


def test_eval_batches_come_from_the_offset(tmp_path, monkeypatch):
    T, B = 8, 2
    monkeypatch.setattr("moestack.data.EVAL_SKIP_TOKENS", 80)   # -> first = 9
    cache = make_cache(tmp_path, n_tokens=(T + 1) * 40, seq_len=T, vocab=10_000)
    src = DiskSource(str(cache), cfg_for(cache, T, B))
    batches = src.eval_batches(2)
    assert len(batches) == 2 and all(x.shape == (B, T) for x, _ in batches)
    first = math.ceil(80 / T) - 1
    assert int(batches[0][0][0, 0]) == first * (T + 1)


def test_eval_raises_when_the_cache_is_too_short(tmp_path):
    cache = make_cache(tmp_path, n_tokens=(8 + 1) * 20, seq_len=8, vocab=10_000)
    src = DiskSource(str(cache), cfg_for(cache, 8, 2))
    with pytest.raises(ValueError, match="Rebuild with more"):
        src.eval_batches(2)


# ── guards ───────────────────────────────────────────────────────────

def test_missing_cache_and_mismatched_settings(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        DiskSource(str(tmp_path / "nope"), cfg_for(tmp_path / "nope"))
    cache = make_cache(tmp_path, n_tokens=(8 + 1) * 20, seq_len=8, vocab=10_000)
    with pytest.raises(ValueError, match="insert_separator"):
        DiskSource(str(cache), cfg_for(cache, 8, 2, insert_separator="none"))


def test_config_requires_cache_dir():
    with pytest.raises(ValueError, match="data.source=disk needs data.cache_dir"):
        Config(data=DataConfig(source="disk"))


# ── end to end ───────────────────────────────────────────────────────

def test_trainer_trains_from_disk(tmp_path, monkeypatch):
    T, B, accum, steps = 32, 4, 2, 6
    # fit() always evaluates at the end; shrink the eval offset to fit a test cache
    monkeypatch.setattr("moestack.data.EVAL_SKIP_TOKENS", 64)
    cache = make_cache(tmp_path, n_tokens=(T + 1) * (B * accum * steps + 16),
                       seq_len=T, vocab=TINY["vocab_size"])
    cfg = tiny_config(tmp_path, train=dict(n_steps=steps, eval_every=0, save_every=0))
    cfg.data.source, cfg.data.cache_dir = "disk", str(cache)
    cfg.data.seq_len, cfg.data.batch_size, cfg.data.grad_accum = T, B, accum
    t = Trainer(cfg, verbose=False)
    res = t.fit()
    assert res["completed_steps"] == steps
    assert res["tokens_seen"] == steps * B * T * accum


def test_prepare_data_roundtrip_writer_to_reader(tmp_path, monkeypatch):
    """build() writes exactly the token stream its loader yielded, and DiskSource
    reads back exactly those sequences. Writer and reader agree on the layout --
    the one thing a synthetic cache cannot check."""
    import data as root_data
    from moestack import prepare_data

    T, B, n_windows = 8, 4, 10
    stream = torch.arange(1, n_windows * (T + 1) + 1)      # 1..N, so 0 means "unwritten"
    windows = stream.view(n_windows, T + 1)

    class FakeDS:
        tokenizer_name, separator_id, vocab_size = "cl100k", 100257, 100277

    class FakeLoader:
        dataset = FakeDS()

        def __iter__(self):
            for i in range(0, n_windows, B):
                w = windows[i:i + B]
                yield w[:, :-1], w[:, 1:]

    monkeypatch.setattr(root_data, "get_dataloader", lambda **kw: FakeLoader())
    out = tmp_path / "built"
    meta = prepare_data.build(out, n_tokens=n_windows * (T + 1), seq_len=T, batch_size=B,
                              num_workers=0, tokenizer="cl100k", insert_separator="before")

    assert meta["n_tokens"] == n_windows * (T + 1)
    on_disk = np.memmap(out / "tokens.u32", dtype=np.uint32, mode="r")
    assert np.array_equal(np.asarray(on_disk), stream.numpy().astype(np.uint32))

    src = DiskSource(str(out), cfg_for(out, T, B))
    inp, tgt = src.batch(0)
    assert torch.equal(inp, windows[:B, :-1])
    assert torch.equal(tgt, windows[:B, 1:])
