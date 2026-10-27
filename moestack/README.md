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
| `expert_accum.py` | token counts → which experts have accumulated a full window and should update now | change the accumulation rule |
| `stats.py` | gini / entropy / spearman / per-expert drift, as pure functions | add a measurement |
| `initialization.py` | depth-scaled Xavier (as before) | change init |
| `optim/builder.py` | tag params by role → `Rule`s → optimizers; LR schedule | change who gets which optimizer/LR/decay |
| `optim/muon.py` | Muon that takes 3D expert tensors directly (batched Newton–Schulz) | change the optimizer math |
| `parallel.py` | distributed init, mesh (`dense`, `expert` views), AC → compile → DDP | add FSDP2 / EP / TP |
| `callbacks.py` | events + `MoELoadStats`, `PerExpertLR`, `ExpertAccumulation`, `ExpertStats`, console, wandb | measure or intervene between steps |
| `trainer.py` | the loop (~100 lines, nothing MoE-specific) | rarely — add an event instead |
| `checkpoint.py` | model + optimizers + data position + RNG + callback state | — |
| `data.py` | `stream` (repo-root `data.py`, unchanged), `disk` (pre-tokenized cache) or `synthetic` | add a data source |
| `prepare_data.py` | writes the `disk` cache once from the stream | — |
| `hf_compat.py` | state-dict mapping to/from HF, `to_hf_model()` for eval/export | — |

## Running

### The token cache

`conf/baseline.yaml` already points at a built cache, so there is nothing to do
unless you need a different one. It exists because a 300-step cluster run spent
~65 % of its wall clock in the HF stream -- 50 min filling the two 10,000-document
shuffle buffers before step 0, ~15 min more for the closing eval, and one outright
DataLoader hang that cost a whole job attempt. None of that was compute.

```yaml
data:
  source: disk                                              # stream | disk | synthetic
  cache_dir: /mloscratch/homes/anagupta/data/dclm-fp-1.2B
```

To build another (CPU job, no GPU needed -- took 34 min: 4.9 min of buffer fill then
29.2 min at 663k tok/s):

```bash
python $CSUB -n prep-data --train --node-type h200 --cpus 8 --memory 64G --command \
  "cd looped-moe && source .venv/bin/activate && mkdir -p logs && \
   python -m moestack.prepare_data --out /mloscratch/homes/$USER/data/dclm-fp-1.2B \
     2>&1 | tee -a logs/prep-data.log"
```

1.16 B tokens = 4.64 GB = **2,211 steps** at 16x16x2048. Past that `DiskSource` wraps
to the start -- a second epoch, silently -- so rebuild with `--tokens 3e9` if you
plan longer runs. Keep `--num-workers` low: each worker holds its own two
10,000-document shuffle buffers in RAM, and 8 of them got the build OOM-killed.

Tokens come from the repo-root loader unmodified, so a `disk` run sees the same
sequences in the same order as a `stream` run. `disk` is also seekable, so resume
seeks instead of replaying the stream.

**The eval slice is the tail of the cache**, not `get_eval_batches`' 10 M-token
offset. That offset is sequence 4,882 at `seq_len=2048`, and a 16x16 run reads 256
sequences a step, so training overran the whole eval set at **step 19** and every
`eval/loss` logged before this change scored memorised text. The tail is held out
for any run that does not wrap the cache, and the condition is checked rather than
assumed: `Trainer` passes the sequences the run will read and `DiskSource.eval_batches`
refuses a slice the run would train on. At 1.16 B tokens with `data.eval_batches=40`
the holdout is sequences 565,489-566,128 and the ceiling is **2,208 steps** at
16x16 on one GPU; ask for more and the run fails at its first eval with the number
to rebuild to. `source=stream` has no tail to retreat to and still reads upstream's
offset, so it warns instead. Train curves are unaffected -- only the eval set moved
-- so `train/lm_loss` still lines up with earlier runs while their `eval/loss` does
not.

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

Unknown keys and invalid enum values fail at startup. The run name spells out arch,
scale, optimizer, aux/z coefficients, lr, seed and `logging.tag`, and carries a
fingerprint of **every** other learning-relevant field that differs from the defaults
(batch size, grad accum, seq len, n_steps, warmup, data source, ...), so cells of a
sweep cannot overwrite each other's checkpoints. The trainer prints the decoded
override list at startup, so the hash is never opaque.

## Running on the EPFL cluster

Two workflows. Use **A** to debug interactively on a GPU; use **B** for anything you
want results from. `csub.py` is the MLO wrapper and is **not in this repo** -- on
this laptop it is at `/Users/ananyagupta/repos/getting-started/csub.py`. The cluster
home is `/mloscratch/homes/$USER`, and the repo lives at
`/mloscratch/homes/$USER/looped-moe` with its venv at `.venv`.

### A. Direct run inside a GPU pod

For a live traceback, a profiler, or poking at a checkpoint. The pod is interactive,
so the run dies when the pod does -- do not use it for a run you care about.

```bash
# 1. from the laptop: start a GPU pod. -t is how long the pod lives before it is
#    torn down (it sleeps for that long), so size it to your debugging session.
CSUB=/Users/ananyagupta/repos/getting-started/csub.py
python $CSUB -n dev-gpu -g 1 --node-type h200 -t 8h

runai list jobs                     # wait for Running (image pull takes a moment)
runai exec dev-gpu -it -- zsh       # 2. shell into it

# 3. inside the pod
cd ~/looped-moe
git pull                            # the pod has whatever the checkout last had
source .venv/bin/activate

# 4. run. `python -m`, not `python moestack/train.py` -- it is a package.
python -m moestack.train data.source=synthetic train.dry_run=true logging.wandb=false
python -m moestack.train loss.aux_coef=0 train.n_steps=300
torchrun --standalone --nproc_per_node=4 -m moestack.train      # multi-GPU, needs -g 4

# 5. from the laptop, when done -- an idle GPU pod still holds the GPU
runai delete job dev-gpu
```

### B. Submit a job from the laptop

`submit_moestack_runs.sh` runs each job from the cluster checkout, and **there is no
branch guard**: if you forget to pull there, you silently train the old code. The job
echoes `commit <sha>` (plus ` (dirty)`) as the first line of its log -- check it
against what you pushed.

```bash
# 1. laptop: commit and push. Nothing reaches the cluster otherwise.
git add -A && git commit -m "..." && git push

# 2. laptop: a CPU pod is enough to drive git and runai (skip if dev-cpu is up)
python /Users/ananyagupta/repos/getting-started/csub.py -n dev-cpu
runai list jobs

# 3. pull inside the cluster -- the step that is easy to forget
runai exec dev-cpu -- bash -c "cd ~/looped-moe && git pull && git log --oneline -1"

# 4. laptop: look before you leap, then submit by exact name
bash submit_moestack_runs.sh --dry parity | grep '>>'
bash submit_moestack_runs.sh --trial parity     # 100 steps, wandb tag "trial"
bash submit_moestack_runs.sh parity             # the real thing
#   no name at all submits EVERY run in the file -- 17 jobs

# 5. monitor
runai list jobs                                 # STATUS: Pending/Running/Failed
runai describe job ms-parity-<batch>            # why it died (e.g. OOMKILLED)
runai logs ms-parity-<batch> | grep -v httpx
runai delete job ms-parity-<batch>
```

Knobs on the submit script: `NODE_TYPE` (h200), `TIME` (16h, enforced by `timeout`
inside the container), `BACKOFF` (4 retries), `BATCH` (the tag that keeps
submissions from colliding). Per-run: `GPUS=n`, `RESUME=1`, `NOTES=".."`.

**Reading the output.** `runai logs` can lag and truncate -- a 2,647-line run showed
1,335 lines through it. The authoritative copy is the tee'd file:

```bash
runai exec dev-cpu -- bash -c \
  "grep -vE 'httpx|HTTP Request' ~/looped-moe/logs/ms-parity-<batch>.log | tail -40"
```

Checkpoints and `<run_name>_results.json` land in `~/looped-moe/checkpoints/`;
per-expert JSONLs sit beside them. wandb groups runs by `Config.condition_name()`,
so seeds of one cell plot together.

**Things that have actually gone wrong here**, so check them first:

| Symptom | Cause |
|---|---|
| `enable_gqa` / `torch.optim.Muon` missing, 26 tests instead of 50 | ran the system python; `source .venv/bin/activate` |
| `OOMKILLED` during `prepare_data` | too many `--num-workers`; each holds 2x10,000 docs in RAM |
| Job runs code you did not write | forgot step 3; check the `commit <sha>` log line |
| Minutes of silence before step 0 | `data.source=stream`; use the `disk` cache |

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
| Resume | replays the stream | replays the stream (`stream`), seeks instantly (`disk`, `synthetic`); restores RNG + callback state | resume is bit-identical to an uninterrupted run (tested) — with one documented exception: under `optim.expert_accum` the in-flight accumulation windows are reset to zero, see [Limits](#limits) |
| Data delivery | HF streaming only: ~50 min of shuffle-buffer filling before step 0, and a stuck worker hangs the run with no timeout | `data.source=disk` memory-maps a cache built once by `prepare_data` | measured: ~65 % of a 300-step run's wall clock was the stream, none of it compute |
| Eval cadence | after step index 500, 1000, … | after 500, 1000, … completed steps | off-by-one cleanup |
| LM loss memory | one-shot `logits.float()` → ~20 GB held, ~33 GB peak at B=16/T=2048/V=100k | `loss.lm_chunk_rows` (4096) chunks it, each chunk's fp32 cast and log-softmax recomputed in backward | same value and gradient; ~10 GB peak, which is what caps batch size at this vocab |
| Non-finite step | applied, then the run raises | dropped (`train/skipped_steps`), run aborts after `train.max_consecutive_skips` in a row | a NaN step poisons the weights *and* the next checkpoint before anything notices |
| Eval scoring | `cross_entropy` without `ignore_index`, divided by `targets.numel()` | the same `lm_cross_entropy_sum` training uses | train and eval cannot drift on which positions count |
| SIGTERM | killed where it lands, losing up to `save_every` steps | checkpoints at the next step boundary and exits; a second signal kills immediately | `submit_moestack_runs.sh` caps each job with `timeout`, so this is the normal way a long run ends. It does **not** help against RunAI preemption: the grace period is 30 s, less than one `regular` step, let alone a 59 GB save — only periodic checkpoints survive that |
| Checkpoint retention | every `_step*.pt` kept forever | `train.keep_last_n` prunes all but the newest N after each save (`best` and the newest are never pruned) | a cadence that survives preemption writes ~20 files a run; at 59 GB each that is 1.2 TB per cell |
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
| `test_ddp.py` | 2-process gloo: DDP grads = single-process full-batch grads; replicas stay identical; expert counts are summed over ranks; per-expert multipliers and EMA are byte-identical on every rank; a never-cleared gradient still accumulates through the reducer's bucket, and zeroing one expert's slice leaves the rest of it alone |
| `test_per_expert_lr.py` | the LR rule vs the original `compute_lr_scales`; `lr_scale` in Muon = per-expert LRs; alpha=0 and splitting groups change nothing bit-for-bit; config validation and run names; resume keeps a warm EMA |
| `test_stats.py` | gini decomposition, Spearman tie handling and NaN, per-layer series, the JSONL's shape and cadence |
| `test_expert_accum.py` | the window rule (threshold, timeout, dead experts, overshoot); Muon's `update_mask` incl. an all-true mask being bit-identical to none; the trainer's new events and accumulator handling; both clocks end to end, and that they agree bit-for-bit at `grad_accum=1`; **resume drops the open windows and keeps the event history**; `keep_last_n` retention |

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

## Expert-local gradient accumulation

Fix the **token budget** per update instead of the clock. The baseline gives every
expert one update per step, built from whatever the router sent it: a big-batch
estimate for a hot expert, a handful of tokens for a cold one, and for a dead one a
full-magnitude step on renormalised stale momentum (`RESULTS.md` §8). Here an
expert keeps its gradient until about `threshold` tokens have reached it, then
updates once and resets — so every update rests on roughly the same number of
tokens, and hot experts simply update more often.

Off by default, and `enabled=false` leaves every path exactly as it was.

```bash
# step clock: updates still land on step boundaries, just not every step
python -m moestack.train loss.aux_coef=0 optim.expert_accum.enabled=true \
    optim.expert_accum.threshold_rel=4 optim.expert_accum.max_window=16

# micro-batch clock: an expert fires the moment its budget is met
python -m moestack.train loss.aux_coef=0 optim.expert_accum.enabled=true \
    optim.expert_accum.clock=microbatch \
    optim.expert_accum.threshold=196608 optim.expert_accum.max_window=256

# the control that is not accumulation: only skip the zero-token experts
python -m moestack.train loss.aux_coef=0 optim.expert_accum.enabled=true \
    optim.expert_accum.threshold=1 optim.expert_accum.max_window=1

bash moestack/submit_expert_accum.sh --dry        # the 10-cell grid
```

| Field | Meaning |
|---|---|
| `clock` | `step` (decide once per optimizer step) or `microbatch` (decide after every backward, so an expert can update several times per batch) |
| `threshold` | Q, in routed tokens |
| `threshold_rel` | Q as a multiple of a balanced expert's load **per tick** — the setting that transfers across scales. At `16×16×2048` with 64 experts and top-6 a balanced expert sees 49,152 tokens per step and 3,072 per micro-batch |
| `max_window` | D, the staleness cap, in ticks. With no threshold at all, D alone closes windows (the fixed-window variant) |
| `grad_scale` | `mean` divides an event by the tokens it was built from; `sum` does not |

### How it is wired

1. **`p.grad` is the window.** The callback registers the expert tensors in
   `trainer.grad_accumulators`, and the trainer stops clearing them between batches
   and leaves them out of `clip_grad_norm_`. So accumulation costs **no extra
   memory** — and clipping an accumulator would not clip a gradient, it would
   shrink a half-filled window by a different factor each step.
2. **`expert_accum.AccumState`** — pure `[L, E]` bookkeeping, decides who fires from
   the same counts the load statistics read.
3. **`Muon` groups carry an `update_mask`** (bool `[E]`). A False slice is skipped
   entirely: no step, no decay, **and no momentum update**. That last part is the
   whole difference from `lr_scale=0`, where the buffer keeps lerping toward the
   zeros an idle expert sees. The masked path gathers the firing slices with
   `index_select`, so a step costs what fires rather than what exists, and an
   all-true mask is bit-identical to no mask.
4. **`Optimizers.step(only="expert")`** lets a mid-batch event move the experts
   while the shared parameters keep the gradient they are still accumulating.

### Rules that are easy to get wrong

- **A window ages from its first token**, not from the last event, so an expert
  quiet for fifty ticks is not fired by the timeout the instant one token arrives.
- **An expert with no tokens never fires** — threshold, timeout or flush.
- **The gradient is divided by the tokens it was built from.** Under Muon the
  overall scale washes out in Newton-Schulz, so this is not about step size: it is
  about `buf.lerp_(grad, 1 - mom)`, which otherwise lets a long window dominate the
  momentum buffer purely by being long.
- **Momentum runs in event time.** An expert firing every fiftieth tick decays its
  buffer once per event. Intended — each expert has its own clock — but a real
  difference from the baseline.
- **A dropped batch throws away every window.** Its gradient is already in every
  accumulator and cannot be subtracted back out. On the micro-batch clock a
  non-finite micro-batch additionally fires nothing, because an expert that has
  stepped cannot be un-stepped.
- **The final flush is not a nicety.** Without it, every expert's last partial
  window is discarded; with a budget larger than a short run ever reaches, that is
  the entire run.

### Limits

**Resume works, but the accumulated gradient is reset to zero.** A resumed run
picks up the weights, the optimizer state, the step counter and the data position
exactly, like any other run — but every expert's *in-flight accumulation window*
is thrown away and restarted empty. An expert that was 80 % of the way to its
token budget when the checkpoint was written begins the resumed run at 0 %.

Why: the window's gradient is `p.grad`, which `checkpoint.py` does not write. It
is not small — the accumulated gradient is one number per expert weight, so at
`regular` scale it is **6,442,450,944 values = 25.8 GB**, as large as the expert
weights themselves. Saving it would take every checkpoint from 59 GB to 84 GB and
the save from ~5 min to ~7 min, which is backwards for a run that is being
preempted. Nor can it be summarised: `pending` is a token *count* (a scalar per
expert, which is why that part is cheap to save), while the gradient is a
6.29 M-dimensional direction per expert, and at Q ≈ 50–100 k tokens against a
2048-dim matrix every window is full rank, so there is nothing to truncate.

So on load the counters are restored and then zeroed, instead of carrying token
counts whose gradient is gone. That second part is the point: restoring
`pending = 90,000` into a process whose `p.grad` is empty would make the next
event divide a 3,000-token gradient by 90,000 and fire a window built from the
wrong tokens, with nothing raising. Zeroing keeps the bookkeeping and the
gradient consistent.

| On resume | What happens to it |
|---|---|
| model weights, Muon momentum, step, data position, RNG | restored exactly |
| `events`, `last_event` (what already fired) | **kept** — else `accum/event_gini` restarts at 0 and reads as a step change |
| `pending`, `age` (the open window) | **reset to 0** |
| `p.grad` (the accumulated gradient) | **gone** — never written, now never claimed to exist |

The cost is bounded: at most `max_window` ticks of partial gradient per expert,
once per resume — well under 1 % of a 2000-step cell even after twenty restarts.
Both halves are reported, **`accum/resumes`** and
**`accum/resume_dropped_windows`**, because a cell resumed twenty times has given
up measurably more accumulation than one resumed never: check those two series
before comparing cells. An uninterrupted run is unaffected — none of this path
runs without a checkpoint to load, so the cells that finish in one attempt stay
bit-identical to how they would have run before resume existed.

This matters because the H200 pool preempts: the first batch of
`submit_expert_accum.sh` lost 72 of 76 attempts, median 1.8 h against the ~18 h a
cell needs. Evicted pods are *deleted and recreated*, so `--backofflimit` does not
cap the churn, and the 30 s grace period is far too short to reach a step boundary
and write a 59 GB file — the graceful SIGTERM path never fired once. Periodic
`save_every` checkpoints are the only thing that survives, which is why the submit
script now sets `save_every=100`, `keep_last_n=2` and `auto_resume=true`.

**`clock=microbatch` is single-process.** The trainer runs all but the last
micro-batch under `no_sync`, so mid-batch an expert's gradient is one rank's alone
and the replicas would diverge silently. Syncing every micro-batch is *correct* —
`test_ddp.py` pins that a never-cleared gradient accumulates properly through the
reducer's bucket — but costs `grad_accum×` the communication, so it is a choice to
make rather than a default. `clock=step` works under DDP unchanged. Expert
parallelism is where this idea actually belongs: each expert's gradient from every
rank's tokens already arrives at its owner through the backward all-to-all, so the
owner can decide and update locally with no extra collective.

**Mutually exclusive with `optim.per_expert_lr`.** They compose fine inside Muon
(mask × multiplier, tested), but the LR rule is keyed on *this step's* token share
while an accumulated update represents a whole window — so the multiplier would be
scaling an update it does not describe. The composition that makes sense is a
per-*event* rule, reading the window's token count at the moment it fires.

### What to watch

`accum/events_per_expert` is the one number that says what the clock is doing: 1.0
is the baseline's cadence, 3.0 means experts update three times as often, 0.2 means
a window spans five steps. Then `accum/tokens_per_event_mean` (the budget actually
delivered) and `accum/timeout_fraction` (high means Q is out of range and the
windows are really fixed-length). At 1.0 and 0 respectively, the cell is a baseline
wearing a different name.

Those plus `accum/event_gini` are the whole `accum/` section. `pending_{mean,max}`,
`age_max`, `tokens_per_event_{min,max}`, `overshoot_mean`, `waiting_fraction` and
`never_fired_fraction` were removed from the dashboard: each is a function of the
per-event `[L, E]` matrices in the JSONL, `age_max` only restated what
`timeout_fraction` shows, and `overshoot_mean` is `tokens_per_event_mean / Q` with
both terms still logged. `accum/discarded_windows` and `accum/nonfinite_microbatches`
are in the run *summary* rather than as series -- see [Measurement](#measurement).

`accum/event_gini` is the tripwire. The scheme can feed a loop: a hot expert
crosses its budget more often, takes more steps, improves, and the router sends it
still more. Read it next to `load/gini` — if update inequality runs away while
routing inequality does not, the extra *steps* are doing it.

`logging.accum_events_every=N` writes `<save_dir>/<run_name>_accum_events.jsonl`:
one record per tick that fired anything, with the full `[L, E]` picture of tokens
per event, which rule closed each window, and what is still pending. The cadence
lives in `logging`, so it never changes a run's name.

## Measurement

**The contract: collect wide, watch narrow.** The `[L, E]` matrices go to a JSONL
every cadence, because they are cheap and you never regret having them. wandb gets
about a dozen scalars, each of which names a decision. A metric earns a panel only
if you can say which decision it changes; otherwise it is recomputed offline from
the matrices. The dashboard is a dozen series, not a catalogue.

`MoELoadStats` runs every step from the exact global counts and emits exactly two:
`load/gini` and `load/dead_expert_fraction`. Gone, and recoverable offline from the
same counts: `gini_alive`, `normalized_entropy`, `max_expert_fraction`,
`max_over_mean`, the worst-layer variants, and the per-layer series that
`logging.load_stats_per_layer` used to switch on (the flag itself is gone).
`max_over_mean` is DeepSeek's MaxVio + 1, so that is the one to recompute when
positioning against published numbers.

Note that `max_expert_fraction` and `max_over_mean` only ever differed by a factor
of `n_experts` — `max/total` against `max/mean` — so they were the same series
logged twice.

With per-expert LR on, `lr_scale/spread_alive` (fastest/slowest working expert, 1.0
= the rule is doing nothing) is the single series: it is what says an alpha sweep is
separating its cells rather than running one experiment repeatedly.
`lr_scale/{min,max,min_alive,max_alive,mean_alive}` and all four `expert_lr/*` are
gone — the latter were these multipliers times the base rate, which is on the
dashboard as `train/lr`. All of this is before Muon's `adjusted_lr`, which adds
`sqrt(max(1, out/in))` per shape, so an expert's `down_proj` and `gate_up_proj`
already differ before any multiplier.

`logging.expert_stats_every=N` writes the wide part. Per measured step it brackets
the optimizer step with a host-side snapshot of the expert tensors and writes
`<save_dir>/<run_name>_expert_stats.jsonl`: per-expert counts, `||Δw||` (total and per
matrix), weight norms and LR multipliers, all `[n_layers][n_experts]`. **It defaults
to 50 rather than 0**, because with the aggregates above removed this file is the
only record of the matrices they were computed from; at 0 that picture is discarded,
not merely unplotted.

The dashboard gets `drift/corr_share_vs_update` (Spearman, ties averaged, NaN rather
than 0 when there is no rank variance), `drift/dead_over_alive` — the
zero-token-expert effect in one number, which `freeze_dead` should drive to 0 — and
`drift/rel_*`. Dropped (non-finite) steps are not recorded: no optimizer step ran,
so the drift would be zero by construction.

`drift/rel` is `||Δw_e|| / ||w_e||`: the quantity Muon actually controls, and the
only one comparable between layers of different width and across training as the
weights grow. It replaces the raw `drift/{mean,max,dead_mean}`. It is reported
through `stats.summarize`, the shared helper for any `[L, E]` quantity, which gives
`mean`, `p10`, `median`, `p90`, `max`, `argmax_layer`, `argmax_expert` and
`outlier_fraction` — seven scalars where the full matrix would be 1,024 series. The
point is the alarm: a mean over experts hides one expert running away, so `max`
pulling clear of `median` is the symptom and the two `argmax_*` integers name the
cell (two ints, not an `"05:17"` string, because wandb plots numbers and files
strings as metadata). `outlier_fraction` is judged per layer, against the median
absolute deviation, falling back to the mean deviation when the MAD is zero — which
is the normal case, since Newton-Schulz drives many update norms to near-identical
values, and a MAD of zero would otherwise switch the alarm off exactly where a lone
outlier is most visible.

**`drift/rel` and `update/rel` are two different measurements.** Both are
`||Δw||/||w||`. `drift/rel` is net displacement over the whole reporting interval,
on the step clock, from `ExpertStats`; `update/rel` is the displacement of a single
*event*, on the event clock, from `ExpertGradStats`, and it is the numerator of
`update/per_token`. The namespace carries the clock: `drift/*` is the interval,
`update/*` is the event. They were both called `update/rel` until the split, with
`ExpertStats` registered last and so winning every reporting step but the first —
read a run logged before the split as `drift/rel_*` wearing the other name, and
expect a discontinuity at its first reporting step, where the per-event value got
through. `CallbackList._fire` now raises if two callbacks write one key in a step.

### Gradient quality: `grad/snr`

The thesis claim in one number. Within a window, expert e's gradient is a sum over
micro-batches; for a cold expert most terms are exactly zero and the rest point in
near-random directions, so the sum is dominated by accident.

    var_e = (1/M) sum_j ||g_e,j||^2 - ||gbar_e||^2
    SNR_e = ||gbar_e|| / sqrt(var_e / M)

**The `/ M` matters.** `||gbar|| / sqrt(var)` -- mean over micro-batches against
their spread, which is how this metric is usually first written down -- is the
*per-sample* ratio, and it is flat in the window size by construction: widen the
window and `gbar` converges to the true mean while `var` converges to the noise
variance, so the ratio converges to a constant. What the optimizer steps on is
`gbar`, whose error is the standard error, and *that* grows as `sqrt(M)` -- the
prediction expert-local accumulation exists to test. The per-sample form also
reports an expert that fired once as `1/sqrt(M-1)`, i.e. getting worse as the
window lengthens around it, when nothing about the expert changed; the standard-error
form reads `sqrt(M/(M-1))` ~ 1.0 for that expert at any M.

Cheap because experts are single 3D `[E, ...]` tensors: a **leaf grad hook** sees
each backward's contribution before it is accumulated into `.grad`, so the two
running `[L, E]` arrays the identity needs come for free -- no differencing of
gradient-sized snapshots. `tests/test_grad_stats.py` pins that the hooks fire
exactly once per backward under non-reentrant activation checkpointing (forward
hooks do not, which is why the router counts come from `RouterOut` instead).

`ExpertGradStats` is only the clock, and the clock is the subtlety: **a window is
measured when it closes, not at the step boundary.** Those coincide for a baseline
run and do not under accumulation, where a window can span several steps or end
mid-step. Measuring at the boundary would divide a gradient accumulated over three
steps by the micro-batch count of the last one. A baseline step is still a window
of `grad_accum` micro-batches, which is what anchors the SNR-vs-window-tokens curve
at the un-accumulated end.

`logging.grad_stats_every=N` (default 50) is the **reporting** cadence only. The
bookkeeping runs continuously, and must: window boundaries belong to
`ExpertAccumulation`, which fires, steps and zeroes gradients on every step whether
or not this callback is reporting. Skipping the reset on off-cadence steps leaves
`sumsq` holding micro-batches whose gradient has been zeroed away, and the ratio is
then between two different intervals -- which reads as SNR ~1e-5 and expert norms
~1e-10, not as an obviously broken number.

**Aggregation is two-stage, and the order matters.** At a reporting step the events
that closed are a biased sample: hot experts fire far more often, so pooling events
directly would weight them fifty to one -- and that weighting moves with any
intervention that changes firing rates, which is the confound the experiment cannot
afford. So: mean within an expert first (one vote each), then `stats.summarize` over
experts. An expert that fired zero times in the interval reports NaN and is masked
out; read `grad/snr` next to `accum/events_per_expert`, which says how many events
backed each mean.

The same two-stage rule applies to `update/per_token` (`rel_update / tokens`, the
quantity that makes windows of different length commensurable) and to
`grad/expert_norm`. One row per event goes to
`<save_dir>/<run_name>_grad_events.jsonl`: `(step, layer, expert, tokens, micro,
snr, grad_norm, rel_update, per_token)`. Path length -- as against the net
displacement `ExpertStats` measures -- is the sum of `rel_update` over an expert's
events, computed from that file.

**Nothing may rescale `p.grad` between the hooks and the read.** SNR compares the
accumulated gradient against the raw per-micro-batch norms the hooks recorded, so a
factor applied to one side only is pure error -- and it is invisible, because the
result is still a plausible-looking number. Two things do this, and both are
global, so neither carries any information about within-window agreement:

- `ExpertAccumulation._arm` divides the firing gradients by their token count
  (`grad_scale="mean"`). The event is therefore opened *inside* `_close_windows`,
  before `_arm` runs.
- `clip_grad_norm_` shrinks the **baseline's** expert gradients by a global factor.
  Only the baseline: once the experts are accumulators, `Trainer._clipped_params`
  excludes them on purpose. So the baseline path divides the clip coefficient back
  out via `open_event(grad_correction=...)`.

Left in, these read as SNR ~29x and ~3.7x low respectively. The test that catches
them is the invariant that a baseline and a `threshold=1` step clock are *the same
measurement* -- both fire every expert every step on the same micro-batches -- so
at step 0, where the weights and data are identical, their SNR must be identical.
It now matches to six figures across both clocks and both `grad_scale` modes.

A baseline step is itself one window of `grad_accum` micro-batches, and its event
carries the step's counts as its token budget, so `update/per_token` is comparable
between a baseline cell and an accumulated one. Without that it would silently be
NaN for every baseline, and comparing the two is the point of the metric.

### Alignment: `compass/*`

Compass (arXiv:2610.04140) sets each expert's Muon step length from how well its
*orthogonalised* update agrees with its current gradient. Muon normalises the
update's singular values, so every expert matrix of a shape moves about the same
distance per step however well-aligned it is -- and alignment degrades exactly when
the tokens routed to an expert change while its momentum still reflects the old
ones. That is this project's premise reached from the optimizer side.

`CompassStats` computes those factors and **does not apply them**; the update stays
plain Muon. `tests/test_compass.py` pins that two runs differing only in whether the
callback is on produce bit-identical weights, so it is safe to switch on mid-grid.

    a_e  = [cos(O_e, G_e)]_+                     expert alignment, eq. 2
    u_e  = a_e / abar,  v_e = min(1, u_e),  c_e = v_e / vbar          family, eq. 3
    q_er = min(1, b_er / bbar_e),  h_er = 1 + lam (q_er - 1)          rows,   eq. 4
    alpha_e = sqrt( sum_r h_er^2 ||O_er||^2 / ||O_e||_F^2 )           radius, eq. 5

with `lam = lr / peak_lr`, so the row factors relax to 1 as the schedule decays. A
*family* is one projection matrix across a layer's experts, so `gate_up_proj` and
`down_proj` are separate families. `compass/scale` is `c_e * alpha_e`: the radius is
in [0, 1] by construction, while `c_e` can exceed 1 because capping `v_e` at 1 pulls
the second mean below one -- which is why the paper's expert scale reaches 1.05-1.10
at the top while its mean sits near 1. Zero-token experts are excluded from the
family means, as in the paper: their gradient is exactly zero, so their "alignment"
is not a measurement.

It pairs with `grad/snr`: SNR asks whether an expert's gradient is trustworthy,
alignment asks whether the step agrees with it, and a cold expert can fail both for
different reasons.

The measurement rides a read-only `probe` hook on `Muon`, which fires inside
`_apply` because the orthogonalised direction exists nowhere else -- it is a local,
consumed by the weight update. `logging.compass_stats_every` is 0 by default (it is
a second cosine over every expert matrix) and requires `optim.name=muon`, which it
raises about rather than silently no-op.

Series are summarized over every expert *matrix*, as the paper does, laid out
`[L, 2E]` so `argmax_layer` is a real layer and `argmax_expert` is
`family * n_experts + expert` (family 0 = `gate_up`, 1 = `down`).

### The run summary

Some numbers are evidence a run is *valid* rather than curves to watch: monotone
counters, constants, and fractions that stay at zero until something has gone wrong.
`callbacks.SUMMARY_ONLY` diverts them to the wandb run summary — one value per run,
sortable in the runs table, no panel: `train/skipped_steps`, `system/gpu_peak_gb`,
`lr_scale/{frozen,clamped}_fraction`, `accum/threshold` (resolved once, so a
constant), `accum/{discarded_windows,nonfinite_microbatches,resumes,resume_dropped_windows}`.

Deleting them would be the wrong trade: `resumes` is why a preempted cell is not
directly comparable to a clean one, and `clamped_fraction` is the warning that
clamping — applied after normalisation — has pushed the alive mean off 1.0, so an
alpha sweep is no longer comparing equal-average cells. You need them to defend a
result; you do not need to watch them arrive.

`eval/` is `loss`, `load/{gini,dead_expert_fraction,maxvio}` and
`routing/{entropy,confidence,margin}` — the same two families as the training-side
`load/*` and `routing/*`, measured on the holdout. The router produces both during
the eval forward anyway, so they cost a softmax per layer and nothing else, and the
holdout is the one slice every cell and every eval step scores *in common*: a cell
whose `load/gini` differs from another's on the training batch has partly been
handed different routing problems, while on the holdout it has not. Counts are
summed over the whole eval set and the moments averaged over its batches, and
nothing is reduced — every rank scores the identical slice, so each already holds
the global number.

Two things they are not. They are not comparable with the training-side series at
the same step: `data.eval_batches=40` is 1.31 M tokens against a step's 524 k, and
both `gini` and `dead_expert_fraction` fall as the token budget rises. And
`dead_expert_fraction` here means "saw no holdout token", which is a statement
about 1.31 M tokens, not about the expert being dead in any permanent sense.

`perplexity` was `exp(loss)`: the same curve on a different axis.

Both cadences live in `logging`, so they never change a run's name or fingerprint.

## Where the next pieces go

- **FSDP2**: `parallelize()` — `fully_shard` each block on the `dense` mesh and the expert
  tensors on `expert["edp"]`, sharding experts **only along dim 0** so every rank holds
  whole expert matrices (keeps Muon's Newton–Schulz and per-expert logic local).
  `checkpoint.py` switches to `torch.distributed.checkpoint` with the same layout.
- **Expert parallelism**: an `AllToAllDispatcher` implementing `dispatch`/`combine`,
  `GroupedExperts(n_local_experts=E/ep)`, and expert grads reduced over `edp` only.
  Test: EP=2 matches EP=1 on loss and grads.
- **Hole-punched holdout** (optional; the eval slice is already held out). The slice
  is now the cache tail, and `DiskSource.eval_batches` raises when a run is long
  enough to reach it -- which answers the objection to the tail, that it fails
  silently on a wrap, by making it fail loudly. What it still costs is the last
  ~2,200 steps' worth of headroom: a longer run has to rebuild the cache rather
  than just run. The alternative is to reserve a hole that `DiskSource.batch` maps
  around, which survives a wrap and costs a branch in the hot index path. Worth
  doing only when runs actually need to go past the cache.

- **Experiments** (tying, routing rules): new `Rule`s, group attributes, or a
  callback/step policy — not edits to `trainer.py`. Per-expert LR and expert-local
  accumulation, above, are the two worked examples: a group attribute, a pure rule
  module and one callback each. Accumulation is the larger of the two and still
  added only three callback events to `trainer.py`, plus one list of parameters
  whose gradient it must not clear or clip — the trainer never learns what the
  accumulation rule is.
