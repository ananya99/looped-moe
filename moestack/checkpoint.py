"""Full-state checkpoints: model, optimizers, trainer counters, data position, RNG,
callback state. A resumed run continues the uninterrupted one exactly (on the
synthetic source this is tested bit-for-bit in tests/test_training.py).

With DDP every rank holds identical weights, so rank 0 writes one torch.save file.
Under FSDP/EP this becomes torch.distributed.checkpoint (DCP) with the same
`state` dict layout -- DCP is resharding-aware, so a checkpoint written at one
mesh can be loaded at another.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def _rng_state():
    s = {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _set_rng_state(s):
    torch.set_rng_state(s["torch"])
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save(path: Path, *, model, optimizers, trainer_state: dict, callbacks, config: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "format": "moestack-v1",
        "model": model.state_dict(),
        "optimizers": optimizers.state_dict(),
        "trainer": trainer_state,
        "callbacks": callbacks.state_dict(),
        "rng": _rng_state(),
        "config": config,
    }, tmp)
    tmp.replace(path)     # atomic: a crash mid-write never leaves a truncated checkpoint


def load(path: Path, *, model, optimizers, callbacks) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt.get("format") == "moestack-v1", f"{path} is not a moestack checkpoint"
    model.load_state_dict(ckpt["model"])
    optimizers.load_state_dict(ckpt["optimizers"])   # torch moves state to each param's device
    callbacks.load_state_dict(ckpt["callbacks"])
    _set_rng_state(ckpt["rng"])
    return ckpt["trainer"]


def _step_files(save_dir: Path, run_name: str) -> list[Path]:
    return sorted(Path(save_dir).glob(f"{run_name}_step*.pt"),
                  key=lambda p: int(p.stem.rsplit("step", 1)[1]))


def latest(save_dir: Path, run_name: str) -> Path | None:
    ckpts = _step_files(save_dir, run_name)
    return ckpts[-1] if ckpts else None


def prune(save_dir: Path, run_name: str, keep: int) -> list[Path]:
    """Delete all but the newest `keep` step checkpoints. Returns what went.

    Only `{run_name}_step*.pt`: `best` is a result, not a restart point, and is
    left alone. `keep < 1` keeps everything, so this is inert until asked for.
    A file that vanishes under us (two ranks, a stale NFS handle) is not an error
    -- the point was for it to be gone.
    """
    if keep < 1:
        return []
    doomed = _step_files(save_dir, run_name)[:-keep]
    for p in doomed:
        p.unlink(missing_ok=True)
    return doomed
