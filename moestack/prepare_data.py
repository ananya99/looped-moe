"""Pre-tokenize the streaming mix into a flat token file on scratch, once.

    python -m moestack.prepare_data --out /mloscratch/homes/anagupta/data/dclm-fp-1.2B

Why: on the cluster a 300-step run spent ~65 % of its wall clock inside the HF
stream -- 50 min filling the two 10,000-document shuffle buffers before the first
batch, ~15 min more for the closing eval, and one outright DataLoader hang that
cost a whole job. None of that is compute. Reading a memory-mapped file instead
makes every run GPU-bound, instantly resumable, reproducible, and immune to Hub
flakiness.

The tokens come out of the repo-root `data.py` loader unchanged, so tokenizer,
separator placement, document order and sequence boundaries are exactly what a
`source=stream` run would have seen. Each yielded (inp, tgt) pair came from one
contiguous window of `seq_len + 1` tokens with no overlap between windows, so
`cat(inp, tgt[-1:])` reconstructs the original stream verbatim.

Sizing: one optimizer step at the baseline config is
batch_size * grad_accum * seq_len = 16 * 16 * 2048 = 524,288 tokens, so a 2,000-step
run needs 1.05 B tokens -- 4.2 GB at uint32 (cl100k's vocab of 100,277 does not fit
in uint16). The default below adds headroom on top of that.

Interrupted builds are not wasted: `meta.json` is rewritten as the file grows and
records how many tokens are actually valid, so a partial cache is still usable for
shorter runs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

# One 2,000-step run at the baseline config, plus ~10 % headroom.
DEFAULT_TOKENS = 1_160_000_000


def build(out: Path, n_tokens: int, seq_len: int, batch_size: int, num_workers: int,
          tokenizer: str, insert_separator: str) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data import get_dataloader          # repo-root data.py, unmodified

    out.mkdir(parents=True, exist_ok=True)
    tok_path, meta_path = out / "tokens.u32", out / "meta.json"

    loader = get_dataloader(seq_len=seq_len, batch_size=batch_size,
                            num_workers=num_workers, tokenizer=tokenizer,
                            insert_separator=insert_separator)
    ds = loader.dataset
    meta = {"format": "moestack-tokens-v1", "dtype": "uint32", "seq_len": seq_len,
            "tokenizer": tokenizer, "tokenizer_name": ds.tokenizer_name,
            "insert_separator": insert_separator, "separator_id": ds.separator_id,
            "vocab_size": ds.vocab_size, "n_tokens": 0,
            "source": "HuggingFaceTB/dclm-edu 75% + HuggingFaceFW/finephrase 25%",
            "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    assert ds.vocab_size < 2 ** 32, ds.vocab_size

    arr = None
    written, t0 = 0, time.time()
    last_t, last_written, fill_secs = 0.0, 0, None
    print(f"filling the shuffle buffers ({num_workers} workers x 2 x 10,000 docs); "
          f"no output until the first batch, ~20 min is normal", flush=True)
    try:
        for inp, tgt in loader:
            if arr is None:
                # Allocated on the first batch, not up front: the buffer fill is
                # where this dies (OOM, a Hub stall), and a build killed there
                # should leave nothing behind rather than a sparse n_tokens*4 file
                # with no meta.json next to it.
                fill_secs = time.time() - t0
                print(f"  first batch after {fill_secs / 60:.1f} min", flush=True)
                arr = np.memmap(tok_path, dtype=np.uint32, mode="w+", shape=(n_tokens,))
                meta_path.write_text(json.dumps(meta, indent=2))
            # One window of seq_len+1 tokens per row, exactly as the stream carved it.
            window = torch.cat([inp, tgt[:, -1:]], dim=1).numpy().astype(np.uint32)
            flat = window.reshape(-1)
            take = min(flat.size, n_tokens - written)
            arr[written:written + take] = flat[:take]
            written += take
            if written >= n_tokens:
                break
            now = time.time()
            if now - last_t > 30:
                # Rate over the last window, not since t0: a cumulative rate is
                # still dragged down by the buffer fill long after streaming starts,
                # and turns a 2-minute remainder into an "eta 753 min".
                rate = (written - last_written) / max(now - max(last_t, t0), 1e-9)
                last_t, last_written = now, written
                eta = (n_tokens - written) / max(rate, 1) / 60
                print(f"  {written / 1e6:,.1f}M / {n_tokens / 1e6:,.0f}M tokens "
                      f"({100 * written / n_tokens:.1f}%)  {rate / 1e3:,.0f}k tok/s  "
                      f"eta {eta:,.0f} min", flush=True)
                meta["n_tokens"] = written
                meta_path.write_text(json.dumps(meta, indent=2))
    except KeyboardInterrupt:
        print("\ninterrupted -- keeping what was written", flush=True)
    finally:
        if arr is not None:
            arr.flush()
            del arr
            # A short build leaves the tail unwritten; drop it so the file is all data.
            os.truncate(tok_path, written * 4)
            meta["n_tokens"] = written
            meta_path.write_text(json.dumps(meta, indent=2))

    secs = time.time() - t0
    stream_secs = max(secs - (fill_secs or 0), 1e-9)
    print(f"\nwrote {written:,} tokens ({written * 4 / 1e9:.2f} GB) to {tok_path}")
    print(f"  shuffle-buffer fill: {(fill_secs or 0) / 60:.1f} min   "
          f"streaming: {stream_secs / 60:.1f} min "
          f"({written / stream_secs / 1e3:,.0f}k tok/s)")
    print(f"  sequences at seq_len={seq_len}: {written // (seq_len + 1):,}")
    print(f"  steps at 16x16x2048:           {written // (seq_len + 1) // 256:,}")
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path, help="cache directory to create")
    p.add_argument("--tokens", type=float, default=DEFAULT_TOKENS,
                   help=f"tokens to write (default {DEFAULT_TOKENS:,} = 2,000 steps + 10%%)")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64, help="build-time only")
    p.add_argument("--num-workers", type=int, default=2,
                   help="build-time only. Each worker holds its own 2x10,000-document "
                        "shuffle buffers in RAM, so this is the memory knob: 8 workers "
                        "OOM-killed a cluster job, 2 completed. Raise it only with "
                        "matching --memory, and watch the first progress line.")
    p.add_argument("--tokenizer", default="cl100k")
    p.add_argument("--insert-separator", default="before", choices=["before", "after", "none"])
    a = p.parse_args()
    build(a.out, int(a.tokens), a.seq_len, a.batch_size, a.num_workers,
          a.tokenizer, a.insert_separator)


if __name__ == "__main__":
    main()
