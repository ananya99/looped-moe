"""Entry point.

    python -m moestack.train                                   # conf/baseline.yaml
    python -m moestack.train loss.aux_coef=0 train.n_steps=2000
    python -m moestack.train data.source=synthetic train.dry_run=true logging.wandb=false
    torchrun --standalone --nproc_per_node=4 -m moestack.train
    python -m moestack.train --cfg job                         # print composed config

Hydra composes conf/baseline.yaml over the typed defaults in config.py, so an
unknown key is an error rather than a silent no-op.
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .config import Config


def to_config(cfg: DictConfig) -> Config:
    merged = OmegaConf.merge(OmegaConf.structured(Config), cfg)
    return OmegaConf.to_object(merged)


@hydra.main(version_base=None, config_path="conf", config_name="baseline")
def main(cfg: DictConfig):
    from .trainer import Trainer
    config = to_config(cfg)
    if int(os.environ.get("RANK", "0")) == 0:
        Path(config.train.save_dir).mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.structured(config),
                       Path(config.train.save_dir) / f"{config.run_name()}_config.yaml")
    trainer = Trainer(config)
    try:
        trainer.fit()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
