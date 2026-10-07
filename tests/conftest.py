import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moestack.config import Config, model_args_for  # noqa: E402

TINY = dict(n_layers=2, vocab_size=256, dim=64, moe_inter_dim=32, n_experts=8, max_seq_len=64)


def tiny_args(arch="deepseek", **kw):
    over = dict(TINY)
    if arch == "qwen3moe":
        over.update(n_heads=4, n_kv_heads=2)
    else:
        over.update(n_heads=4, n_kv_heads=4)
    over.update(top_k=2)
    over.update(kw)
    return model_args_for(arch, "tiny", **over)


def tiny_config(tmp_path, **sections) -> Config:
    cfg = Config()
    cfg.model.overrides = dict(TINY, n_heads=4, n_kv_heads=4, top_k=2)
    cfg.data.source = "synthetic"
    cfg.data.seq_len, cfg.data.batch_size, cfg.data.grad_accum = 32, 4, 2
    cfg.data.eval_batches = 2
    cfg.train.n_steps, cfg.train.log_every = 6, 1000
    cfg.train.eval_every = cfg.train.save_every = 0
    cfg.train.save_dir = str(tmp_path)
    cfg.train.device = "cpu"
    cfg.optim.warmup_steps = 2
    cfg.logging.wandb = False
    for section, values in sections.items():
        for k, v in values.items():
            setattr(getattr(cfg, section), k, v)
    return cfg


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: compiles with torch.compile (deselect with -m 'not slow')")
