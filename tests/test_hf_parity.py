"""Our model computes the same function as the HF reference classes the original
trainer used (OlmoeForCausalLM / Qwen3MoeForCausalLM with grouped_mm experts), and
our loss terms equal the original trainer's patched router loss."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from moestack.hf_compat import from_hf_state_dict, to_hf_model, to_hf_state_dict
from moestack.initialization import init_weights
from moestack.losses import lm_cross_entropy, load_balance_loss, router_z_loss
from moestack.model import Transformer

from conftest import tiny_args

ARCHS = ["deepseek", "olmoe", "qwen3moe"]


def original_patched_router_loss(gate_logits, top_k, aux_coef, z_coef, carrier=0.01):
    """Verbatim logic of imbalance_train.install_router_loss_patch, times the carrier
    HF multiplies it by -> the effective router loss added to lm loss."""
    aux_scale, z_scale = aux_coef / carrier, z_coef / carrier
    total, n = 0.0, 0
    for layer_logits in gate_logits:
        n_exp = layer_logits.size(-1)
        flat = layer_logits.view(-1, n_exp)
        f32 = flat.float()
        probs = torch.softmax(f32, dim=-1).type_as(flat)
        z = torch.mean(torch.logsumexp(f32, dim=-1) ** 2) * z_scale
        _, selected = torch.topk(probs, top_k, dim=-1)
        mask = torch.nn.functional.one_hot(selected, n_exp)
        tokens_per_expert = mask.sum(dim=1).float().mean(dim=0) / top_k
        balance = (tokens_per_expert * probs.mean(dim=0)).sum() * n_exp
        total = total + aux_scale * balance + z
        n += 1
    return carrier * total / n


def _pair(arch):
    torch.manual_seed(0)
    args = tiny_args(arch, experts_impl="grouped_mm")
    ours = Transformer(args)
    init_weights(ours)
    hf = to_hf_model(ours, arch).eval()
    hf.config._attn_implementation = "eager"
    return ours.eval(), hf


@pytest.mark.parametrize("arch", ARCHS)
def test_state_dict_roundtrip(arch):
    ours, _ = _pair(arch)
    sd = ours.state_dict()
    back = from_hf_state_dict(to_hf_state_dict(sd))
    assert back.keys() == sd.keys()


@pytest.mark.parametrize("arch", ARCHS)
def test_logits_and_router_logits_match_hf(arch):
    ours, hf = _pair(arch)
    x = torch.randint(0, ours.args.vocab_size, (2, 24))
    with torch.no_grad():
        a = ours(x)
        b = hf(input_ids=x, output_router_logits=True)
    torch.testing.assert_close(a.logits, b.logits, atol=2e-5, rtol=1e-4)
    assert len(b.router_logits) == len(a.router_outs)
    for r, hl in zip(a.router_outs, b.router_logits):
        torch.testing.assert_close(r.logits, hl, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("arch", ARCHS)
def test_losses_match_original_trainer(arch):
    ours, hf = _pair(arch)
    x = torch.randint(0, ours.args.vocab_size, (2, 24))
    with torch.no_grad():
        a = ours(x)
        b = hf(input_ids=x, labels=x, output_router_logits=True)
    # lm: HF shifts labels internally, i.e. scores logits[:, :-1] against x[:, 1:]
    from transformers.loss.loss_utils import ForCausalLMLoss
    ours_lm = lm_cross_entropy(a.logits[:, :-1], x[:, 1:])
    torch.testing.assert_close(ours_lm, ForCausalLMLoss(b.logits, x, ours.args.vocab_size),
                               atol=1e-5, rtol=1e-5)
    # router loss: our terms combined with coefficients == original patched function
    for aux_coef, z_coef in [(0.01, 1e-4), (0.0, 1e-4), (0.01, 0.0)]:
        ours_router = (aux_coef * torch.stack([load_balance_loss(r) for r in a.router_outs]).mean()
                       + z_coef * torch.stack([router_z_loss(r) for r in a.router_outs]).mean())
        orig = original_patched_router_loss(b.router_logits, ours.args.top_k, aux_coef, z_coef)
        torch.testing.assert_close(ours_router, orig, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("arch", ARCHS)
def test_gradients_match_hf(arch):
    ours, hf = _pair(arch)
    ours.train(), hf.train()
    x = torch.randint(0, ours.args.vocab_size, (2, 24))
    lm = lm_cross_entropy(ours(x).logits[:, :-1], x[:, 1:])
    lm.backward()
    hf(input_ids=x, labels=x, output_router_logits=False).loss.backward()
    hf_grads = {k: p.grad for k, p in hf.named_parameters()}
    ours_grads = to_hf_state_dict({n: p.grad.clone() for n, p in ours.named_parameters()})
    pad = hf.config.pad_token_id
    if pad is not None:
        # Known divergence, intentional: HF builds nn.Embedding(..., padding_idx=pad_token_id)
        # and OlmoeConfig defaults pad_token_id=1, so the original trainer never updated
        # the embedding of token 1 (cl100k: '"') -- only weight decay touched it. There is
        # no padding in packed pretraining, so moestack trains every row.
        assert hf_grads["model.embed_tokens.weight"][pad].abs().sum() == 0
        ours_grads["model.embed_tokens.weight"][pad] = 0
    for k, g in ours_grads.items():
        torch.testing.assert_close(g, hf_grads[k], atol=5e-5, rtol=1e-3, msg=k)
