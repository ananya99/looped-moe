# moestack — modular MoE pretraining baseline

A refactor of the **baseline** path of `imbalance_train.py` / `moe_train.py` (untied,
no per-expert LR, no tying) into small modules with one job each. The old scripts are
untouched and remain the reference; `tests/test_hf_parity.py` proves the new model
computes the same function as the HF classes they used.

```
config ──► mesh ──► model (pure math) ──► parallelize ──► optimizers (rules) ──► loop + callbacks
                                                                       ▲
                                    data (micro-batch iterator) ───────┘     checkpoint
```

| Module | Job | Change it when you want to… |
|---|---|---|
| `config.py` | Typed config tree + arch presets (`deepseek`/`olmoe`/`qwen3moe` × `tiny`/`small`/`regular`) | add an option |
| `model/moe.py` | `Router` → `Dispatcher` → `GroupedExperts`, returns `RouterOut` | change routing (Router), expert parallelism / comms (Dispatcher), kernels (Experts) |
| `model/layers.py`, `model/transformer.py` | RMSNorm, RoPE, attention, blocks | change the dense architecture |
| `losses.py` | `lm + aux_coef·balance + z_coef·z`, each term reported separately | add a loss term |
| `initialization.py` | depth-scaled Xavier (as before) | change init |
| `optim/builder.py` | tag params by role → `Rule`s → optimizers; LR schedule | change who gets which optimizer/LR/decay |
| `optim/muon.py` | Muon that takes 3D expert tensors directly (batched Newton–Schulz) | change the optimizer math |
| `parallel.py` | distributed init, mesh (`dense`, `expert` views), AC → compile → DDP | add FSDP2 / EP / TP |
| `callbacks.py` | events + `MoELoadStats`, console, wandb | measure or intervene between steps |
| `trainer.py` | the loop (~100 lines, nothing MoE-specific) | rarely — add an event instead |
| `checkpoint.py` | model + optimizers + data position + RNG + callback state | — |
| `data.py` | `stream` (repo-root `data.py`, unchanged) or `synthetic` | add a data source |
| `hf_compat.py` | state-dict mapping to/from HF, `to_hf_model()` for eval/export | — |

## Running

```bash
python -m moestack.train                                      # conf/baseline.yaml
python -m moestack.train loss.aux_coef=0 train.seed=1         # any field, Hydra syntax
python -m moestack.train data.source=synthetic train.dry_run=true logging.wandb=false
torchrun --standalone --nproc_per_node=4 -m moestack.train
python -m moestack.train --cfg job                            # print the composed config

python -m pytest tests -q                                     # all tests, CPU, ~35 s
python -m pytest tests -q -m "not slow"                       # skip compile + 2-process DDP
```

Unknown keys and invalid enum values fail at startup. The run name encodes arch,
scale, optimizer, aux/z coefficients, lr, seed and `logging.tag`, so cells of a sweep
never overwrite each other's checkpoints.

## What is the same as `imbalance_train.py` (baseline cell)

- Architectures and widths (the `get_*_config` presets), including the `deepseek`
  misnomer (OLMoE layout, top-6).
- Forward function — logits and router logits match `OlmoeForCausalLM` /
  `Qwen3MoeForCausalLM` with `grouped_mm` experts (tested, fp32, all three archs);
  experts run in the weights' dtype (fp32) as HF's grouped_mm path did.
- Router loss: `aux_coef · mean_l(balance_l) + z_coef · mean_l(z_l)` — tested equal to
  the patched `load_balancing_loss_func` times its carrier, for several coefficient pairs.
- Depth-scaled Xavier init; Muon (lr, momentum 0.95, Nesterov, wd 0.1) for attention
  and expert matrices; AdamW (lr×0.1, wd 0.01, β=(0.9, 0.95)) for router, embeddings,
  head, norms; warmup + cosine to 0.1×; grad clip 1.0; bf16 autocast on CUDA.
- Defaults in `conf/baseline.yaml`: deepseek-tiny, 16 × 16 micro-batches × 2048, 2000 steps.

## What is intentionally different

| | Original | moestack | Why |
|---|---|---|---|
| Embedding of token id 1 | never updated: HF builds `nn.Embedding(padding_idx=pad_token_id)` and `OlmoeConfig` defaults `pad_token_id=1` (cl100k `"`), so only weight decay touched that row | trained like every other row | no padding in packed pretraining; this was a silent bug (affects `deepseek`/`olmoe`, not `qwen3moe`) |
| LM loss positions | `labels=input_ids` → HF scores T−1 positions per sequence | scores all T using `targets` | uses the token `data.py` already provides; ≈0.05 % more targets |
| Balance-loss top-k | re-runs top-k on probs cast to the autocast dtype | uses the router's own fp32 top-k | identical in fp32; avoids picking a different expert on bf16 near-ties |
| Muon on experts | one 2D proxy `Parameter` per expert + grad copy every step | batched Newton–Schulz on the 3D tensor | same math per matrix (tested vs `torch.optim.Muon`, ≤1 bf16 ulp), no proxies or Python loop over experts |
| Resume | replays the stream | replays the stream (`stream`), seeks instantly (`synthetic`); restores RNG + callback state | resume is bit-identical to an uninterrupted run (tested) |
| Eval cadence | after step index 500, 1000, … | after 500, 1000, … completed steps | off-by-one cleanup |

Checkpoints are `moestack-v1` files (not HF state dicts). For `eval_downstream.py`, convert with
`hf_compat.to_hf_model(model, arch).save_pretrained(path)`.

## Tests (`tests/`)

| File | What it pins down |
|---|---|
| `test_hf_parity.py` | logits, router logits, LM loss, router loss and gradients vs HF / the original patch (3 archs) |
| `test_moe.py` | MoE = naive per-token definition; grouped_mm = loop (fwd + bwd); dispatch conserves tokens; combine with identity experts; permutation equivariance; every parameter receives gradient |
| `test_losses_and_optim.py` | balance loss = 1 when balanced, ≈E when collapsed; batched Muon = `torch.optim.Muon` per slice (incl. zero-grad expert); role → optimizer table; LR curve = original `get_lr` |
| `test_training.py` | overfit one batch; grad accumulation = full batch (per-token terms); activation checkpointing is transparent and does not double-count routing; trainer end to end; **resume = uninterrupted run, bit for bit**; EP>1 fails loudly; compile = eager |
| `test_ddp.py` | 2-process gloo: DDP grads = single-process full-batch grads; replicas stay identical; expert counts are summed over ranks |

Not covered here (needs a GPU): bf16 autocast numerics, CUDA `grouped_mm`, NCCL. First
thing to run on the cluster: a short `deepseek-tiny` run next to an `imbalance_train.py`
run with the same seed and data, comparing `train/lm_loss`.

## Where the next pieces go

- **FSDP2**: `parallelize()` — `fully_shard` each block on the `dense` mesh and the expert
  tensors on `expert["edp"]`, sharding experts **only along dim 0** so every rank holds
  whole expert matrices (keeps Muon's Newton–Schulz and per-expert logic local).
  `checkpoint.py` switches to `torch.distributed.checkpoint` with the same layout.
- **Expert parallelism**: an `AllToAllDispatcher` implementing `dispatch`/`combine`,
  `GroupedExperts(n_local_experts=E/ep)`, and expert grads reduced over `edp` only.
  Test: EP=2 matches EP=1 on loss and grads.
- **Experiments** (per-expert LR, expert-local accumulation, tying): new `Rule`s, group
  attributes, or a callback/step policy — not edits to `trainer.py`.
