# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research code for the paper *"Tying the Loop: Tied Expert Layers in Mixture-of-Experts
Language Models"*, plus a follow-up experiment on expert-load imbalance.

**Expert tying** = sharing expert FFN weight tensors across consecutive transformer layers
while routers, attention, and norms stay layer-specific. There is no linter config and no
package metadata — this is a set of standalone training scripts.

Terminology warning: "tied/shared experts" here never means DeepSeek-style always-on shared
experts. It means `layers[i].mlp.experts.gate_up_proj is layers[j].mlp.experts.gate_up_proj`
— literally the same `nn.Parameter` object, so autograd sums gradients across use sites.

## Three independent stacks — they share nothing but `data.py`

| Stack | Files | What it is |
|---|---|---|
| **Section 3 ablation** | `train.py` + `model.py` | Hand-written depth-32 transformer with *looped* topologies. Small-scale component ablation (which components can be tied). Not a production MoE. |
| **Section 4 main** | `moe_train.py` | Real MoE training on HuggingFace `transformers` reference models (`OlmoeForCausalLM`, `Qwen3MoeForCausalLM`). This is where the paper's headline results come from. |
| **Expert-imbalance experiment** | `imbalance_train.py` + `expert_stats.py` | The `moe_train.py` pipeline with the tying machinery stripped out, plus per-expert token measurement and per-expert learning rates. |
| Eval | `eval_downstream.py` | lm-eval-harness wrapper; reads checkpoints from either MoE trainer. |

`model.py` is imported only by `train.py`. `moe_train.py` never touches it, and
`imbalance_train.py` imports neither — it is a standalone copy of the `moe_train.py`
pipeline, not a wrapper around it. When a request mentions "the model", ask which stack: they
have separate config systems, separate `build_optimizer` implementations, and separate
LR-scaling flags.

## Commands

```bash
pip install -r requirements.txt   # pinned torch 2.12 / transformers 5.9; lm-eval from git

# --- Section 4, paper results (moe_train.py) ---
python moe_train.py --arch deepseek --scale tiny --tie-group-size 1 --dry-run   # 100-step smoke test
python moe_train.py --arch olmoe --scale tiny --tie-group-size 4 --optimizer muon \
    --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --n-steps 20000 --auto-resume
torchrun --standalone --nproc_per_node=4 moe_train.py --arch qwen3moe --tie-group-size 4 ...

# --- Expert-imbalance experiment (imbalance_train.py) ---
python imbalance_train.py --arch deepseek --scale tiny --aux-loss-coef 0 --n-steps 2000
python imbalance_train.py --arch deepseek --scale tiny --aux-loss-coef 0 \
    --per-expert-lr-alpha 1.0 --freeze-dead-experts
bash submit_imbalance_runs.sh              # the 8-run matrix (cluster only)
python test_expert_stats.py                # CPU tests for the measurement logic

# --- Section 3 ablation (train.py) ---
python train.py --dry-run                                        # 100 steps
python train.py --variant onegroup_32_experttie --config small --optimizer muon --tied-lr-mode sqrt
python train.py --experiment exp1 --config small                 # a whole variant group
bash submit_ablation_runs.sh                                     # full 43-run grid (cluster only)

# --- Eval ---
python eval_downstream.py --ckpt checkpoints/<run>_step5000.pt --tie-group-size 4 \
    --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa --num-fewshot 3
```

`QUICKSTART.md`, `submit_ablation_runs.sh` and `submit_imbalance_runs.sh` are lists of
`csub.py ...` cluster submissions. **`csub.py` is not in this repo** — it is the EPFL cluster
wrapper, locally at `/Users/ananyagupta/repos/getting-started/csub.py`. Strip the
`csub.py -n ... --command "..."` envelope to get the runnable `python`/`torchrun` command.

`--tie-group-size` must match the value used at training time when evaluating, or the
checkpoint's state dict will not load. Checkpoints from `imbalance_train.py` are always
untied, so they need `--tie-group-size 1`.

## Cluster access

Runs, logs, checkpoints and the per-expert JSONLs live on the EPFL RunAI cluster, not in this
repo. Home there is `/mloscratch/homes/anagupta`; this repo is at
`/mloscratch/homes/anagupta/looped-moe`.

Inspecting artefacts needs no GPU — start a CPU-only pod:

```bash
python /Users/ananyagupta/repos/getting-started/csub.py -n dev-cpu

runai describe job dev-cpu        # check job status
runai logs dev-cpu                # view logs
runai exec dev-cpu -it -- zsh     # connect to the pod
runai delete job dev-cpu          # delete it when done
```

The same `runai` commands take any job name, so they also work on the training jobs submitted
by `submit_imbalance_runs.sh` / `submit_ablation_runs.sh`.

---

# Original code

## Section 4: `moe_train.py`

Single ~1400-line `train()` function. Order of operations inside it matters:

1. **Build HF model** with `config._experts_implementation = "grouped_mm"`, so expert weights
   are **3D** `[n_experts, out, in]` tensors. `verify_grouped_mm()` prints which dispatch path
   is live.
2. **Router-loss monkeypatch** (`_install_router_loss_patch`) replaces HF's
   `load_balancing_loss_func` in every module where it lives, adding a router z-loss
   (Zoph et al.) to the standard Shazeer/Fedus balancing loss. The z-loss is pre-divided by
   `router_aux_loss_coef` because HF post-multiplies the whole return value. It asserts at
   least one patch site was found, so an HF upgrade fails loudly rather than silently
   dropping the z-loss. This runs for *every* run, tied or not.
3. **`--expand-tied-experts N`** ("transplant") builds a throwaway model with `num_experts=N`
   and swaps its `mlp` into the *middle* layers only. Used for iso-parameter controls — both
   width expansion at `g>1` and narrow untied baselines at `g=1`.
4. **Depth-scaled Xavier re-init** replaces HF's `trunc_normal(0.02)` with
   `xavier_uniform(gain=1/sqrt(3*depth))`, matching `model.py`.
5. **`tie_expert_layers()`** runs before `.to(device)`. It chunks `layers[skip_first : n-skip_last]`
   into groups of `group_size` and aliases each follower's `gate_up_proj`/`down_proj` onto the
   group leader's. It asserts afterwards that experts are shared *and* routers are not.
6. **`build_optimizer()`** — see below. Built before `torch.compile` so param ids match.

### The 2D-proxy trick (most important detail in the file)

Muon cannot step on a 3D expert tensor. `build_optimizer` creates **one 2D proxy
`nn.Parameter` per expert** sharing storage with `p.data[i]`, hands those to Muon, and
`DualOptimizer._sync_expert_grads()` copies `param_3d.grad[i]` into each proxy before every
step. The 3D tensors themselves are registered with no optimizer, so `zero_grad` clears them
explicitly.

Consequence: **every expert is already its own optimizer parameter**, so per-expert LR/WD
experiments need no model surgery — just regroup the proxies and set `pg["lr"]`. This is what
`imbalance_train.py` builds on.

Parameter routing (weight decay is **hardcoded** here; the `weight_decay` threaded in from
`TRAIN_CONFIGS` is dead):

| Parameter | Optimizer | LR | WD |
|---|---|---|---|
| Expert FFN 3D (`mlp.experts.*`) | Muon, via proxies | `lr` | 0.1 |
| Attention / other 2D | Muon | `lr` | 0.1 |
| Router (`mlp.gate.weight`) | AdamW | `lr * 0.1` | 0.01 |
| Embeddings, `lm_head`, all 1D | AdamW | `lr * 0.1` | 0.01 |

### Run names and checkpoints

`run_name` is derived from flags: `{arch}-g{N}` or `{arch}-notie`, plus `-we{N}` for
`--expand-tied-experts` and `-small`/`-tiny` for `--scale`. It drives both wandb and
`checkpoints/{run_name}_{step|best}.pt`. Changing a flag that is *not* in the name (e.g.
`--tied-lr-divisor`) will silently collide with a previous run's checkpoints —
`--auto-resume` globs `{run_name}_step*.pt` and picks the highest step.

## Section 3: `model.py` + `train.py`

`ModelConfig.topology` is a list of `(n_layers, n_loops)` tuples: `n_layers` unique blocks
run `n_loops` times in sequence. All named variants hold effective depth at 32, e.g.
`[(2,1), (2,14), (2,1)]`. Sharing mode is set by three booleans, and the variant-name suffix
encodes which:

- `alltie` — everything shared across loop iterations
- `attntie` — per-loop routers; FFN and attention shared
- `lora` — per-loop routers plus a per-loop LoRA adapter on attention
- `experttie` — only the expert FFN shared (`per_loop_attn=True`)
- `ffnonly` — the dense-model equivalent of `experttie`

`LoopedMoETransformer.make_variant(name, base_cfg_kwargs)` is the single source of truth:
`VARIANT_TABLE` at the bottom of `model.py` defines every variant, and `train.py`'s
`VARIANTS`/`EXP*` lists must stay in sync with it. Size presets live in `train.py`'s `CONFIGS`
dict (`tiny`/`small`/`small-coarse`/`2b`/`2b-coarse`/`small-dense`/`2b-dense`); `--config`
selects one, `VARIANT_TRAIN_OVERRIDES_*` adjusts batch size per variant to hold tokens/step
constant.

`train.py` buckets Muon params by the `n_loops` of the group they belong to and applies
`--tied-lr-mode {none,linear,sqrt}` per bucket. This is a *different mechanism* from
`moe_train.py`'s scalar `--tied-lr-divisor`; do not port flags between the two files.

## `data.py`

Streaming 75:25 interleave of `HuggingFaceTB/dclm-edu` and `HuggingFaceFW/finephrase`,
packed into fixed-length sequences and sharded across DDP ranks × dataloader workers.
Defaults: cl100k tokenizer (vocab 100277, matching the hardcoded model configs) with one
separator token *before* each document. `get_eval_batches()` skips the first 10M tokens and
sets `do_shard=False` so every rank evaluates on an identical holdout. Training and eval must
be given the same `tokenizer`/`insert_separator`/`separator_id`.

---

# Expert-imbalance experiment

New files, added on top of the original code without modifying any of it.

The question: **how does the number of tokens routed to an expert influence the learning of
that expert's weights and of its router row, and can the learning rate be varied per expert
based on it?** Imbalance is largest early in training, so short runs are the right instrument.

## Why this needs its own trainer

Muon's Newton-Schulz orthogonalisation normalises the update's singular values, so it is
invariant to gradient *scale*. It is **not** invariant to gradient *rank*: an expert that saw
`n` tokens has a rank-≤`n` gradient, so its update norm goes as `sqrt(min(n, d))` (measured in
`RESULTS.md` §8 — the original "step size is independent of token count" premise was wrong).
Two consequences shape the whole design:

- Scaling an expert's **gradient** does nothing; only its **learning rate** survives
  normalisation. Hence per-expert parameter groups rather than gradient hooks.
- A **zero-token expert still moves**: its gradient is exactly zero, but Newton-Schulz
  renormalises its decaying momentum buffer back to unit norm, so it takes a full-magnitude
  step in a stale direction while weight decay shrinks it every step.

## `imbalance_train.py`

A standalone copy of the `moe_train.py` pipeline. Dropped, because all of it is inert at
`--tie-group-size 1`: `tie_expert_layers()`, the flags `--tie-group-size`,
`--tie-skip-first/last`, `--expand-tied-experts`, `--tied-lr-divisor`, the
`cross_loop_agreement`/`routing_diversity` metrics, and the `router/*` dashboard metrics
(which were top-1, last-micro-batch and rank-0 — superseded by the exact `load/*` series).

Kept, because it applies to the baseline and not just to tying: the router z-loss patch, the
depth-scaled Xavier re-init, the Muon 2D-proxy split, DDP, bf16 autocast, gradient
checkpointing, and checkpoint resume.

Two deliberate differences from `moe_train.py`:

- **Every expert is always its own Muon parameter group**, so per-expert LR is a dict
  assignment rather than a code path. At `--per-expert-lr-alpha 0` without
  `--freeze-dead-experts` all multipliers are exactly 1.0 and this reduces to the plain
  baseline. Consequence: the **optimizer state** in its checkpoints is not interchangeable
  with `moe_train.py`'s, though the model weights are.
- **`run_name` encodes every flag that defines a matrix cell** — `deepseek-tiny-aux0-a1`,
  `deepseek-tiny-aux0-freeze`, `deepseek-tiny-aux0-aneg0.5` — so runs cannot silently
  overwrite each other's checkpoints the way an arch-and-tying-only name does.

`AUX_CARRIER = 0.01` is the subtlest thing in the file. HF computes
`loss = lm_loss + config.router_aux_loss_coef * f(gate_logits)`, so that coefficient is held
at a fixed non-zero carrier and each term inside `f` is pre-divided by it. That makes
`--aux-loss-coef` and `--z-loss-coef` independent, which is what allows load balancing to be
switched off (`--aux-loss-coef 0`, the imbalanced regime) while the router z-loss stays on.

## `expert_stats.py`

- `ExpertLoadTracker` hooks every `mlp.gate` and accumulates **exact top-k counts over the
  full global batch** — all micro-steps, all DDP ranks. It must be armed around the forward
  and disarmed before backward: with non-reentrant gradient checkpointing the forward is
  replayed inside backward and the hooks would double every count.
- Counts are `all_reduce`d **before** any learning rate is derived from them. Per-expert LRs
  must be identical on every rank or the replicas silently diverge.
- `compute_lr_scales` implements `lr_e ∝ s_e**alpha` plus the freeze-dead-experts control.
  Multipliers are normalised to average 1.0 over alive experts by default, so changing alpha
  redistributes learning rate without also changing the average rate. Clamping is asymmetric
  (floor 1e-3, ceiling 10) because a small multiplier is merely slow whereas a large one can
  destabilise Muon. Deadness is keyed on **raw per-step counts** even when the rule itself
  runs on EMA counts — otherwise an expert with EMA history would never be recognised as dead.
- Per-expert series are too wide for wandb, so wandb gets the aggregates (`load/gini`,
  `load/dead_expert_fraction`, `load/corr_share_vs_update`) and the full joint distribution
  goes to `<save_dir>/<run_name>_expert_stats.jsonl`.
- Per-expert LR requires `--optimizer muon`: the 2D proxies are what make each expert
  individually addressable. Under AdamW the experts stay as single 3D tensors, and the flags
  raise rather than silently no-op.

## Running the experiment

`submit_imbalance_runs.sh` is the 8-run matrix (**~4.3–5 h each** on one H200, not the ~1 h
its header claims; peak VRAM 93.7 GB): Section A measures (aux coefficient
0.01 / 0.001 / 0), Section B intervenes in the imbalanced regime (freeze-dead, alpha 0.5, 1,
-0.5), Section C controls with AdamW.

**Compare these runs on `train/lm_loss`, never `train/loss`** — they differ in their aux
coefficient, so only the pure cross-entropy is comparable across cells.

The `imb-adamw` cell OOMs at the default batch — lower batch size or enable gradient
checkpointing before rerunning it. wandb project: `lord-of-the-strings/moe-expert-imbalance`
(the crashed and re-run `aneg0.5` share a name — filter on run state).

## State of this work

Docs: **`RESULTS.md`** (batch-1 analysis, monitoring playbook, ranked next steps) is the
current source of truth; **`TRAINING_STACK.md`** is a walkthrough of the optimizer/stats code
whose Muon step-size claims (its §2, §7, §8) are corrected by `RESULTS.md` §9. Both are
untracked in git.

- Batch 1 ran 2026-09-29. Headlines (n=1 seed): `--aux-loss-coef 0` gives gini 0.73 / 16%
  dead experts; `alpha=0.5` matches the aux-loss baseline on lm_loss *and* roughly halves
  imbalance; `alpha=-0.5` is clearly harmful; `--freeze-dead-experts` is a null result.
- Gaps: `imb-adamw` crashed (OOM), `imb-mid-a0` (aux 0.001) never launched, no seeds.
- The per-expert JSONLs (`checkpoints/<run_name>/<run_name>_expert_stats.jsonl`) are still on
  the cluster (see [Cluster access](#cluster-access)); there is **no analysis script in the
  repo** — `RESULTS.md` was built from the wandb API (snippet at its end).
- `test_expert_stats.py` covers the statistics and LR-rule logic on CPU (36 checks; needs only
  torch, no GPU and no transformers).

## Gotchas

- **`--tie-group-size` defaults to 4 in `moe_train.py`.** Pass `1` explicitly for an untied
  baseline or you will silently train a tied model. `imbalance_train.py` has no such flag and
  is always untied.
- `--tied-lr-divisor` is meant to be sqrt(g): `1.0` for g=1, `1.41` for g=2, `2.0` for g=4.
  It defaults to `1.0` and is not validated against `--tie-group-size`.
- Muon's `_adjust_lr` multiplies the LR by `sqrt(max(1, A/B))` **per parameter shape**. At tiny
  scale the `gate_up` slices get `1.0` but the `down_proj` slices get `sqrt(2)`, so an expert's
  two matrices already learn at different rates before any per-expert multiplier is applied.
- `README.md` documents the eval flag as `--checkpoint`; the actual flag is `--ckpt`.
- `get_deepseek_config()` returns an `OlmoeConfig` with `num_experts_per_tok=6` — no MLA, no
  fine-grained experts, no shared expert. `--arch deepseek` is not a real DeepSeekMoE.
- `--scale {regular,small,tiny}` changes widths only; layer counts are fixed per arch
  (olmoe/deepseek 16, qwen3moe 28).
- `eval_downstream.py` infers `scale` from substrings in the checkpoint *path*, so renaming a
  checkpoint file can change how it is loaded.
- Resuming replays the stream micro-batch by micro-batch (`step × grad_accum` iterations),
  which is slower than restarting a short run from scratch.
- `data.py` streams from the HF Hub, so runs need network, and `get_eval_batches()` burns 10M
  tokens before the first step.
