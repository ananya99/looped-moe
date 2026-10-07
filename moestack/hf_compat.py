"""Conversion to/from HF `OlmoeForCausalLM` / `Qwen3MoeForCausalLM`.

Used by tests/test_hf_parity.py to prove the model computes the same function as
the HF reference the original trainer used, and usable to export a checkpoint for
`eval_downstream.py` / lm-eval (`to_hf_model(...).save_pretrained(...)`).
"""

from __future__ import annotations

import re

import torch

from .config import ModelArgs

_RENAMES = [  # (ours, hf) -- per-layer keys use {i}
    ("tok_embeddings.weight", "model.embed_tokens.weight"),
    ("norm.weight", "model.norm.weight"),
    ("output.weight", "lm_head.weight"),
    ("layers.{i}.attention_norm.weight", "model.layers.{i}.input_layernorm.weight"),
    ("layers.{i}.ffn_norm.weight", "model.layers.{i}.post_attention_layernorm.weight"),
    ("layers.{i}.attention.q_proj.weight", "model.layers.{i}.self_attn.q_proj.weight"),
    ("layers.{i}.attention.k_proj.weight", "model.layers.{i}.self_attn.k_proj.weight"),
    ("layers.{i}.attention.v_proj.weight", "model.layers.{i}.self_attn.v_proj.weight"),
    ("layers.{i}.attention.o_proj.weight", "model.layers.{i}.self_attn.o_proj.weight"),
    ("layers.{i}.attention.q_norm.weight", "model.layers.{i}.self_attn.q_norm.weight"),
    ("layers.{i}.attention.k_norm.weight", "model.layers.{i}.self_attn.k_norm.weight"),
    ("layers.{i}.moe.router.weight", "model.layers.{i}.mlp.gate.weight"),
    ("layers.{i}.moe.experts.gate_up_proj", "model.layers.{i}.mlp.experts.gate_up_proj"),
    ("layers.{i}.moe.experts.down_proj", "model.layers.{i}.mlp.experts.down_proj"),
]


def _to_regex(template: str) -> re.Pattern:
    return re.compile("^" + re.escape(template).replace(r"\{i\}", r"(\d+)") + "$")


_OURS_TO_HF = [(_to_regex(a), b) for a, b in _RENAMES]
_HF_TO_OURS = [(_to_regex(b), a) for a, b in _RENAMES]


def _convert(sd: dict, table) -> dict:
    out = {}
    for k, v in sd.items():
        for pat, tmpl in table:
            m = pat.match(k)
            if m:
                out[tmpl.format(i=m.group(1)) if m.groups() else tmpl] = v
                break
        else:
            raise KeyError(f"no mapping for {k}")
    return out


def to_hf_state_dict(sd: dict) -> dict:
    return _convert(sd, _OURS_TO_HF)


def from_hf_state_dict(sd: dict) -> dict:
    return _convert(sd, _HF_TO_OURS)


def hf_config(args: ModelArgs, arch: str):
    common = dict(vocab_size=args.vocab_size, hidden_size=args.dim,
                  num_hidden_layers=args.n_layers, num_attention_heads=args.n_heads,
                  num_key_value_heads=args.n_kv_heads, num_experts=args.n_experts,
                  num_experts_per_tok=args.top_k, norm_topk_prob=args.norm_topk_prob,
                  max_position_embeddings=args.max_seq_len, hidden_act="silu",
                  rms_norm_eps=args.norm_eps, tie_word_embeddings=False, use_cache=False,
                  rope_theta=args.rope_theta)
    if arch == "qwen3moe":
        from transformers import Qwen3MoeConfig
        assert args.qk_norm == "head" and args.clip_qkv is None
        return Qwen3MoeConfig(moe_intermediate_size=args.moe_inter_dim, head_dim=args.hd,
                              decoder_sparse_step=1, **common)
    from transformers import OlmoeConfig
    assert args.qk_norm == "full"
    return OlmoeConfig(intermediate_size=args.moe_inter_dim, clip_qkv=args.clip_qkv, **common)


def to_hf_model(model, arch: str):
    cfg = hf_config(model.args, arch)
    cfg._experts_implementation = "grouped_mm"
    if arch == "qwen3moe":
        from transformers import Qwen3MoeForCausalLM as Cls
    else:
        from transformers import OlmoeForCausalLM as Cls
    hf = Cls(cfg)
    missing, unexpected = hf.load_state_dict(to_hf_state_dict(model.state_dict()), strict=False)
    missing = [k for k in missing if "rotary" not in k]
    assert not missing and not unexpected, (missing, unexpected)
    return hf
