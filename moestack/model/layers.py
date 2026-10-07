"""Dense building blocks: RMSNorm, rotary embeddings, attention.

Numerics deliberately follow HF's OLMoE / Qwen3-MoE modules (norm computed in fp32,
cos/sin computed in fp32 then cast) so that `tests/test_hf_parity.py` can demand
near-exact agreement. Nothing in here knows about distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelArgs


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)

    def reset_parameters(self):
        nn.init.ones_(self.weight)


class RotaryEmbedding(nn.Module):
    """Standard (non-interleaved, rotate-half) RoPE."""

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device, dtype):
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, enabled=False):
            freqs = torch.outer(pos, self.inv_freq.to(device).float())
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(dtype), emb.sin().to(dtype)     # [T, head_dim]


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q, k: [B, H, T, hd]; cos, sin: [T, hd]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class Attention(nn.Module):
    """Causal GQA with optional QK-norm (full-width as in OLMoE, per-head as in
    Qwen3) and optional QKV clipping (OLMoE)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.hd = args.n_heads, args.n_kv_heads, args.hd
        self.clip_qkv = args.clip_qkv
        self.qk_norm = args.qk_norm
        self.q_proj = nn.Linear(args.dim, self.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(args.dim, self.n_kv_heads * self.hd, bias=False)
        self.v_proj = nn.Linear(args.dim, self.n_kv_heads * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.hd, args.dim, bias=False)
        if args.qk_norm == "full":
            self.q_norm = RMSNorm(self.n_heads * self.hd, args.norm_eps)
            self.k_norm = RMSNorm(self.n_kv_heads * self.hd, args.norm_eps)
        elif args.qk_norm == "head":
            self.q_norm = RMSNorm(self.hd, args.norm_eps)
            self.k_norm = RMSNorm(self.hd, args.norm_eps)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.qk_norm == "full":
            q, k = self.q_norm(q), self.k_norm(k)
        if self.clip_qkv is not None:
            q, k, v = (t.clamp(-self.clip_qkv, self.clip_qkv) for t in (q, k, v))
        q = q.view(B, T, self.n_heads, self.hd)
        k = k.view(B, T, self.n_kv_heads, self.hd)
        v = v.view(B, T, self.n_kv_heads, self.hd)
        if self.qk_norm == "head":
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, k, *rope)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=self.n_kv_heads != self.n_heads)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))
