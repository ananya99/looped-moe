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


def _missing_torch_features():
    """Probe for the features moestack needs, rather than parsing versions."""
    import torch.nn.functional as F
    missing = []
    try:
        q = torch.zeros(1, 2, 2, 4)
        F.scaled_dot_product_attention(q, q[:, :1], q[:, :1], enable_gqa=True)
    except TypeError:
        missing.append("F.scaled_dot_product_attention(enable_gqa=...)  [torch >= 2.5]")
    except Exception:
        pass        # some other complaint: the kwarg exists, which is all we asked
    if not hasattr(torch.optim, "Muon"):
        missing.append("torch.optim.Muon  (reference for the batched-Muon tests)")
    return missing


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: compiles with torch.compile (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "grouped_mm: needs F.grouped_mm (the fused expert GEMM)")
    # One clear message beats sixteen tracebacks. The usual cause is running the
    # system python instead of the project venv -- e.g. pytest in a fresh cluster
    # pod, where /usr/local/lib/.../torch is years older than requirements.txt.
    missing = _missing_torch_features()
    if missing:
        raise pytest.UsageError(
            f"this interpreter cannot run the moestack tests.\n"
            f"  torch {torch.__version__} from {torch.__file__}\n"
            f"  missing: " + "\n           ".join(missing) + "\n"
            f"requirements.txt pins torch==2.12.0 / transformers==5.9.0.\n"
            f"On the cluster:  cd ~/looped-moe && source .venv/bin/activate && "
            f"python -m pytest tests -q")
