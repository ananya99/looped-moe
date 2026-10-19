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
| `expert_lr.py` | token counts → one LR multiplier per expert (`s_e**alpha`, freeze-dead, EMA) | change the per-expert LR rule |
| `stats.py` | gini / entropy / spearman / per-expert drift, as pure functions | add a measurement |
| `initialization.py` | depth-scaled Xavier (as before) | change init |
| `optim/builder.py` | tag params by role → `Rule`s → optimizers; LR schedule | change who gets which optimizer/LR/decay |
| `optim/muon.py` | Muon that takes 3D expert tensors directly (batched Newton–Schulz) | change the optimizer math |
| `parallel.py` | distributed init, mesh (`dense`, `expert` views), AC → compile → DDP | add FSDP2 / EP / TP |
| `callbacks.py` | events + `MoELoadStats`, `PerExpertLR`, `ExpertStats`, console, wandb | measure or intervene between steps |
| `trainer.py` | the loop (~100 lines, nothing MoE-specific) | rarely — add an event instead |
| `checkpoint.py` | model + optimizers + data position + RNG + callback state | — |
| `data.py` | `stream` (repo-root `data.py`, unchanged), `disk` (pre-tokenized cache) or `synthetic` | add a data source |
| `prepare_data.py` | writes the `disk` cache once from the stream | — |
| `hf_compat.py` | state-dict mapping to/from HF, `to_hf_model()` for eval/export | — |

## Running

### Build the token cache first

A 300-step cluster run spent ~65 % of its wall clock in the HF stream: 50 min
filling the two 10,000-document shuffle buffers before step 0, ~15 min more for the
closing eval, and one outright DataLoader hang that cost a whole job. Pre-tokenize
once and every run after that is GPU-bound:

```bash
python -m moestack.prepare_data --out /mloscratch/homes/$USER/data/dclm-fp-1.2B
python -m moestack.train data.source=disk data.cache_dir=/mloscratch/homes/$USER/data/dclm-fp-1.2B
```

The default size is 1.16 B tokens = 4.6 GB = a 2,000-step run at the baseline config
plus 10 % headroom. Tokens come from the repo-root loader unmodified, so a `disk` run
sees the same sequences in the same order as a `stream` run; the eval slice
reproduces `get_eval_batches`' 10 M-token offset exactly. `disk` is also seekable, so
resume is instant instead of replaying the stream.

```bash
python -m moestack.train                                      # conf/baseline.yaml
python -m moestack.train loss.aux_coef=0 train.seed=1         # any field, Hydra syntax
python -m moestack.train loss.aux_coef=0 optim.per_expert_lr.alpha=0.5 \
    logging.expert_stats_every=50                             # per-expert LR + measurement
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
| Resume | replays the stream | replays the stream (`stream`), seeks instantly (`disk`, `synthetic`); restores RNG + callback state | resume is bit-identical to an uninterrupted run (tested) |
| Data delivery | HF streaming only: ~50 min of shuffle-buffer filling before step 0, and a stuck worker hangs the run with no timeout | `data.source=disk` memory-maps a cache built once by `prepare_data` | measured: ~65 % of a 300-step run's wall clock was the stream, none of it compute |
| Eval cadence | after step index 500, 1000, … | after 500, 1000, … completed steps | off-by-one cleanup |
| LM loss memory | one-shot `logits.float()` → ~20 GB held, ~33 GB peak at B=16/T=2048/V=100k | `loss.lm_chunk_rows` (4096) chunks it, each chunk's fp32 cast and log-softmax recomputed in backward | same value and gradient; ~10 GB peak, which is what caps batch size at this vocab |
| Non-finite step | applied, then the run raises | dropped (`train/skipped_steps`), run aborts after `train.max_consecutive_skips` in a row | a NaN step poisons the weights *and* the next checkpoint before anything notices |
| Eval scoring | `cross_entropy` without `ignore_index`, divided by `targets.numel()` | the same `lm_cross_entropy_sum` training uses | train and eval cannot drift on which positions count |
| SIGTERM | killed where it lands, losing up to `save_every` steps | checkpoints at the next step boundary and exits; a second signal kills immediately | `submit_moestack_runs.sh` caps each job with `timeout`, so this is the normal way a long run ends |
| Run name | a curated list of flags; changing batch size, grad accum, seq len or n_steps silently reused another run's checkpoints | `condition_name()` carries a fingerprint of **every** learning-relevant field that differs from the defaults | new options are part of a run's identity by default and must be excluded on purpose |

Checkpoints are `moestack-v1` files (not HF state dicts). For `eval_downstream.py`, convert with
`hf_compat.to_hf_model(model, arch).save_pretrained(path)`.

## Tests (`tests/`)

| File | What it pins down |
|---|---|
| `test_hf_parity.py` | logits, router logits, LM loss, router loss and gradients vs HF / the original patch (3 archs) |
| `test_moe.py` | MoE = naive per-token definition; grouped_mm = loop (fwd + bwd); dispatch conserves tokens; combine with identity experts; permutation equivariance; every parameter receives gradient |
| `test_losses_and_optim.py` | balance loss = 1 when balanced, ≈E when collapsed; batched Muon = `torch.optim.Muon` per slice (incl. zero-grad expert); role → optimizer table; LR curve = original `get_lr` |
| `test_training.py` | overfit one batch; grad accumulation = full batch (per-token terms); activation checkpointing is transparent and does not double-count routing; trainer end to end; **resume = uninterrupted run, bit for bit**; EP>1 fails loudly; compile = eager |
| `test_ddp.py` | 2-process gloo: DDP grads = single-process full-batch grads; replicas stay identical; expert counts are summed over ranks; per-expert multipliers and EMA are byte-identical on every rank |
| `test_per_expert_lr.py` | the LR rule vs the original `compute_lr_scales`; `lr_scale` in Muon = per-expert LRs; alpha=0 and splitting groups change nothing bit-for-bit; config validation and run names; resume keeps a warm EMA |
| `test_stats.py` | gini decomposition, Spearman tie handling and NaN, per-layer series, the JSONL's shape and cadence |

Not covered here (needs a GPU): bf16 autocast numerics, CUDA `grouped_mm`, NCCL. First
thing to run on the cluster: a short `deepseek-tiny` run next to an `imbalance_train.py`
run with the same seed and data, comparing `train/lm_loss`.

## Per-expert learning rates

The expert-imbalance intervention from `imbalance_train.py`, rebuilt on the seams.
Off by default: `alpha=0` with `freeze_dead=false` makes every multiplier exactly
1.0 and the run is bit-identical to one built before the option existed (tested).

```bash
python -m moestack.train loss.aux_coef=0 optim.per_expert_lr.alpha=0.5
python -m moestack.train loss.aux_coef=0 optim.per_expert_lr.freeze_dead=true
python -m moestack.train loss.aux_coef=0 optim.per_expert_lr.alpha=-0.5   # -> ...-aneg0.5-...
```

| Field | Meaning |
|---|---|
| `alpha` | `lr_e ∝ s_e**alpha`, `s_e = n_e / mean(n)`. 0 = uniform, 1 = token-proportional, <0 = compensate cold experts upward |
| `freeze_dead` | zero-token experts get multiplier 0 — no step **and** no weight decay |
| `ema` | smoothing of the counts the rule reads (0.9 ≈ 10 steps, as `imbalance_train.py` defaulted to and the batch-1 runs used); deadness always uses the raw per-step counts |
| `normalize` | rescale to mean 1.0 over the experts that got tokens, so alpha redistributes rate without changing the average |
| `clamp_min` / `clamp_max` | asymmetric on purpose: a floor only makes an expert slow, a ceiling can destabilise Muon |

How it is wired, in three pieces that each do one thing:

1. `expert_lr.lr_scales_from_counts` — pure `[L, E] → [L, E]`, checked against the
   original `expert_stats.compute_lr_scales` row by row.
2. `Muon` groups carry an `lr_scale` vector that scales the step and the decoupled
   decay per expert. No 2D proxies: the multiplier applies inside the batched
   Newton-Schulz step. `lr_scale=None` is the unscaled path, byte for byte.
3. `PerExpertLR` reads `state.expert_counts` — already summed over micro-batches
   and all-reduced, so every rank derives identical rates — and writes the scales
   onto the groups. Its EMA is checkpointed.

The multipliers are not gradients, so DDP never synchronises them; what makes the
ranks agree is that the counts are all-reduced *before* any callback runs. A 2-rank
test asserts the multipliers, the EMA and the replicas' weights are byte-identical.

Enabling the rule changes the optimizer's param-group layout, so a checkpoint
written with it cannot be resumed into a baseline run (and vice versa). The run
names differ anyway; an explicit `train.resume` path pointed across the two now
fails with a message naming the cause.

`optim.name=adamw` with the rule on fails at startup: there the experts are single
3D tensors with one scalar LR and nothing is individually addressable.

Scope: expert FFN weights only. Per-expert rates on the matching router row
(AdamW, not Muon) are a second rule, deliberately not included.

## Measurement

`MoELoadStats` runs every step from the exact global counts: `load/gini`,
`load/gini_alive`, `load/normalized_entropy`, `load/dead_expert_fraction`,
`load/max_over_mean`, worst-layer variants, and one series per layer
(`logging.load_stats_per_layer=false` to drop those).

`gini = dead_fraction + (1 - dead_fraction) · gini_alive` exactly (tested), which
is why both are logged: a rule that revives dead experts and one that evens out the
working experts both lower `gini`, and only the split says which happened.

With per-expert LR on, `lr_scale/*` give each expert's rate **relative** to the base
LR (`spread_alive` = fastest/slowest working expert) and `expert_lr/*` the same in LR
units, comparable with `train/lr`. Those are before Muon's `adjusted_lr`, which adds
`sqrt(max(1, out/in))` per shape — so an expert's `down_proj` and `gate_up_proj`
already differ before any multiplier.

`logging.expert_stats_every=N` adds the part wandb cannot hold. Per step measured it
brackets the optimizer step with a host-side snapshot of the expert tensors and writes
`<save_dir>/<run_name>_expert_stats.jsonl`: per-expert counts, `||Δw||` (total and per
matrix), weight norms and LR multipliers, all `[n_layers][n_experts]`. The dashboard
gets `drift/corr_share_vs_update` (Spearman, ties averaged, NaN rather than 0 when
there is no rank variance) and `drift/dead_over_alive` — the zero-token-expert effect
in one number, which `freeze_dead` should drive to 0. Dropped (non-finite) steps are
not recorded: no optimizer step ran, so the drift would be zero by construction.

The cadence lives in `logging`, so it never changes a run's name or fingerprint.

## Where the next pieces go

- **FSDP2**: `parallelize()` — `fully_shard` each block on the `dense` mesh and the expert
  tensors on `expert["edp"]`, sharding experts **only along dim 0** so every rank holds
  whole expert matrices (keeps Muon's Newton–Schulz and per-expert logic local).
  `checkpoint.py` switches to `torch.distributed.checkpoint` with the same layout.
- **Expert parallelism**: an `AllToAllDispatcher` implementing `dispatch`/`combine`,
  `GroupedExperts(n_local_experts=E/ep)`, and expert grads reduced over `edp` only.
  Test: EP=2 matches EP=1 on loss and grads.
- **Experiments** (expert-local accumulation, tying): new `Rule`s, group attributes, or
  a callback/step policy — not edits to `trainer.py`. Per-expert LR, above, is the
  worked example: a `Rule` flag, a group attribute and one callback, with `trainer.py`
  changed by two lines that only decide whether to build them.
