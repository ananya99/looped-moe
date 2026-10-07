"""Decoder-only MoE transformer. Single-device math only: no torch.distributed,
no autocast, no checkpointing. `parallel.py` applies those from the outside."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import ModelArgs
from .layers import Attention, RMSNorm, RotaryEmbedding
from .moe import MoE, RouterOut


@dataclass
class ModelOutput:
    logits: torch.Tensor                 # [B, T, V]
    router_outs: list[RouterOut]         # one per MoE layer, in layer order


class Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.attention = Attention(args)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.moe = MoE(args)

    def forward(self, x, rope):
        x = x + self.attention(self.attention_norm(x), rope)
        h, r = self.moe(self.ffn_norm(x))
        return x + h, r


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.rope = RotaryEmbedding(args.hd, args.rope_theta)
        self.layers = nn.ModuleList(Block(args, i) for i in range(args.n_layers))
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> ModelOutput:
        h = self.tok_embeddings(input_ids)
        rope = self.rope(input_ids.size(1), h.device, h.dtype)
        router_outs = []
        for layer in self.layers:
            h, r = layer(h, rope)
            router_outs.append(r)
        return ModelOutput(self.output(self.norm(h)), router_outs)

    def moe_layers(self):
        return [layer.moe for layer in self.layers]
