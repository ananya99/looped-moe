"""Weight init used by the original trainer: depth-scaled Xavier for every matrix.

`imbalance_train.reinit_weights_depth_scaled` applied xavier_uniform(gain=1/sqrt(3*L))
to every 2D parameter (embeddings, attention, routers, lm_head) and to every
per-expert [out, in] slice of the 3D expert tensors, and left 1D parameters (norm
weights = 1) alone. Same here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .model import Transformer, RMSNorm


@torch.no_grad()
def init_weights(model: Transformer) -> None:
    gain = 1.0 / math.sqrt(3 * model.args.n_layers)
    for m in model.modules():
        if isinstance(m, RMSNorm):
            m.reset_parameters()
    for p in model.parameters():
        if p.dim() == 2:
            nn.init.xavier_uniform_(p, gain=gain)
        elif p.dim() == 3:
            for i in range(p.size(0)):
                nn.init.xavier_uniform_(p[i], gain=gain)
