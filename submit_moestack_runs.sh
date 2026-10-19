#!/usr/bin/env bash
# ===================================================================
# Submit moestack runs to the MLO RunAI cluster (via csub.py)
# ===================================================================
#
#   bash submit_moestack_runs.sh                      # submit EVERY run below (17 jobs!)
#   bash submit_moestack_runs.sh parity               # submit just this one (exact name, no regex)
#   bash submit_moestack_runs.sh --dry | grep '>>'    # print the commands, submit nothing
#   bash submit_moestack_runs.sh --trial parity       # submit it, but 100 steps, tagged "trial"
#   bash submit_moestack_runs.sh --batch b1013-0930 aux0-s0   # reuse a batch tag (to resume)
#
# --trial appends train.n_steps=$TRIAL_STEPS and logging.tags=[trial]; Hydra's last value wins,
# so it overrides whatever the run line asked for. Use it to smoke-test a cell before
# committing hours to it, and filter the wandb tag "trial" out of real results.
#
# Env knobs: NODE_TYPE (h200)  BACKOFF (4)  TIME (16h)  BATCH (b<MMDD-HHMM>); BATCH
# also takes a flag, and the flag wins over the env var.
# Paths to csub.py / the repo / the venv are hardcoded just below -- edit them there.
#
# Each run is one line:  [GPUS=n] [NOTES=".."] [RESUME=1] [TIME=..] run <name> <hydra overrides...>
#   GPUS     1 -> python -m moestack.train;  n>1 -> torchrun --nproc_per_node=n (DDP)
#   NOTES    one-line purpose, goes to wandb notes
#   RESUME   1 -> train.auto_resume=true (worth it for long runs; short runs restart faster)
#   TIME     wall-clock cap for this run, e.g. 12h / 45m / 2d6h30m
#   overrides are Hydra key=value pairs over moestack/conf/baseline.yaml,
#   e.g. loss.aux_coef=0 train.seed=1 data.grad_accum=4
#   <name> is what you pass as the argument above. The k8s job is ms-<name>-$BATCH with
#   '.' rewritten to '-', so run name aux0.01-s0 -> job ms-aux0-01-s0-b1013-0930.
#
# Every job:
#   - runs from $REPO_DIR in your scratch home, against whatever that checkout holds.
#     There is no branch guard: push, pull there, then submit. The commit SHA (plus
#     " (dirty)") is echoed into the log as the only record of what actually ran.
#   - tees stdout/stderr to $REPO_DIR/logs/<job>.log;
#   - gets logging.tag=$BATCH, so run names, checkpoints, wandb names and logs of
#     different submissions never collide. The wandb run name and checkpoint prefix
#     are derived by Config.run_name() from the config, not from <name>:
#     {arch}-{scale}-{optim}-aux{..}-z{..}-lr{..}-{fingerprint}-s{seed}-{tag}, where
#     {fingerprint} is 6 hex chars over EVERY other learning-relevant field that
#     differs from the defaults -- batch size, grad accum, seq len, n_steps, warmup,
#     data source, per-expert LR, ... -- so two cells cannot share a checkpoint
#     prefix. The trainer prints the decoded override list at startup. The wandb
#     group is Config.condition_name() -- the same minus seed and tag -- so seeds of
#     one cell plot together as mean and spread;
#   - is capped at $TIME by `timeout` in the container, because runai has no wall-clock
#     flag and csub drops -t whenever --command is given. Keep $TIME well above the
#     expected runtime: it is a hang guard (HF stream stall, NCCL deadlock), not a
#     budget. On expiry the job exits 124 and runai retries it up to $BACKOFF times, so
#     $TIME is per attempt -- only runs with RESUME=1 progress across retries.
#
# Compare runs on train/lm_loss, not train/loss, whenever aux/z coefficients differ.
#
# DATA: conf/baseline.yaml defaults to the pre-tokenized cache (data.source=disk),
# which holds 1.16B tokens = 2,211 steps at 16x16x2048 = 566,129 sequences. A run
# needing more than that wraps to the start of the cache and silently trains a
# second epoch. Sequences a run needs = batch_size * grad_accum * n_steps * GPUS.
# Everything below fits except long-aux0.01 (20k steps = 5.1M sequences, 9x over);
# give that one data.source=stream or build a bigger cache before trusting it.
# ===================================================================

set -euo pipefail

# ---- fixed paths (edit them here; not overridable from the environment) -------
CSUB=/Users/ananyagupta/repos/getting-started/csub.py  # EPFL cluster wrapper, on this laptop
REPO_DIR=looped-moe                          # relative to the job's working dir (scratch home)
VENV=.venv                                   # relative to REPO_DIR

# ---- run knobs (override any of these from the environment) -------------------
NODE_TYPE="${NODE_TYPE:-h200}"
BACKOFF="${BACKOFF:-4}"                      # runai retries before a job is marked Failed
TIME="${TIME:-16h}"
DRY="${DRY:-0}"                              # or pass --dry
TRIAL=0                                      # 1 once --trial is passed
TRIAL_STEPS=100                              # what --trial shortens every run to
BATCH="${BATCH:-b$(date +%m%d-%H%M)}"

while [ $# -gt 0 ]; do                       # flags first, then an optional run name
    case "$1" in
        --dry|-n)  DRY=1;   shift ;;
        --trial)   TRIAL=1; shift ;;
        --batch)   shift; BATCH="${1:-}"; shift || true
                   [ -n "$BATCH" ] || { echo "--batch needs a tag, e.g. --batch b1013-0930" >&2; exit 1; } ;;
        --batch=*) BATCH="${1#--batch=}"; shift
                   [ -n "$BATCH" ] || { echo "--batch needs a tag, e.g. --batch=b1013-0930" >&2; exit 1; } ;;
        -h|--help) sed -n '2,48p' "$0"; exit 0 ;;
        -*)        echo "unknown flag: $1 (try --help)" >&2; exit 1 ;;
        *)         break ;;
    esac
done
ONLY="${1:-}"                                # exact run name; empty submits every run
# Hydra overrides applied to every run in this file.
COMMON=()

# ---- machinery -------------------------------------------------------------------
n_matched=0

to_seconds () {   # 12h | 45m | 2d6h30m | 3600 -> seconds (same grammar as csub.py)
    local s="$1" total=0 n u
    [[ "$s" =~ ^([0-9]+d)?([0-9]+h)?([0-9]+m)?([0-9]+s?)?$ ]] || {
        echo "Invalid duration '$s'. Use formats like 12h, 45m, 2d6h30m." >&2; return 1; }
    while [[ "$s" =~ ^([0-9]+)([dhms]?) ]]; do
        n="${BASH_REMATCH[1]}"; u="${BASH_REMATCH[2]}"
        case "$u" in
            d) total=$((total + n * 86400)) ;;
            h) total=$((total + n * 3600))  ;;
            m) total=$((total + n * 60))    ;;
            *) total=$((total + n))         ;;
        esac
        s="${s#"${BASH_REMATCH[0]}"}"
    done
    echo "$total"
}

run () {   # [GPUS=n] [NOTES=..] [RESUME=1] run <name> <overrides...>
    local base="$1"; shift
    local gpus="${GPUS:-1}" notes="${NOTES:-}" resume="${RESUME:-0}" time="${TIME}"
    local secs; secs="$(to_seconds "$time")"   # validated here so a typo fails before submission
    # k8s job names: lowercase alphanumerics and '-', <= 63 chars
    local job
    job="$(echo "ms-$base-$BATCH" | tr '[:upper:]' '[:lower:]' | tr '_.' '--' | tr -cd 'a-z0-9-' | cut -c1-63)"

    # ${arr[@]+"${arr[@]}"}: macOS's bash 3.2 treats an empty array as unbound under set -u
    local overrides=(${COMMON[@]+"${COMMON[@]}"} ${1+"$@"} "logging.tag=$BATCH")
    [ "$resume" = 1 ] && overrides+=("train.auto_resume=true")
    [ -n "$notes" ] && overrides+=("logging.notes='$notes'")
    # Appended last so Hydra's last-value-wins overrides the run line's own n_steps.
    [ "$TRIAL" = 1 ] && overrides+=("train.n_steps=$TRIAL_STEPS" "logging.tags=[trial]")

    if [ -n "$ONLY" ] && [ "$ONLY" != "$base" ]; then
        return
    fi
    n_matched=$((n_matched + 1))

    local launcher="python -m moestack.train"
    [ "$gpus" -gt 1 ] && launcher="torchrun --standalone --nproc_per_node=$gpus -m moestack.train"

    # %q-quote each override so spaces/quotes survive the remote shell.
    local args="" o
    for o in "${overrides[@]}"; do args+=" $(printf %q "$o")"; done

    local cmd="set -o pipefail && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
cd $REPO_DIR && source $VENV/bin/activate && mkdir -p logs && \
echo \"[$job] commit \$(git rev-parse --short HEAD)\$(git diff --quiet || echo ' (dirty)')\" | tee -a logs/$job.log && \
timeout -k 120 $secs $launcher$args 2>&1 | tee -a logs/$job.log; rc=\$?; \
[ \$rc -eq 124 ] && echo \"[$job] stopped: $time wall-clock budget exhausted\" | tee -a logs/$job.log; \
exit \$rc"

    # no -t: csub only uses --time to build its keep-alive sleep, which --command replaces.
    local flags=(-n "$job" -g "$gpus" --node-type "$NODE_TYPE" --train --backofflimit "$BACKOFF")
    [ "$gpus" -gt 1 ] && flags+=(--large-shm)
    [ "$DRY" = 1 ] && flags+=(--dry)

    echo ">> $job  ($gpus GPU)"
    python3 "$CSUB" "${flags[@]}" --command "$cmd"
}

trial_note=""
[ "$TRIAL" = 1 ] && trial_note="  trial=$TRIAL_STEPS steps"
echo "Submitting batch $BATCH  ${ONLY:+only='$ONLY'  }dry=$DRY$trial_note"

# =====================================================================================
# A — sanity: does the refactor behave on GPU?
# =====================================================================================

# A.0  ~2 min: bf16 autocast + CUDA grouped_mm + checkpoint/eval path, no network.
NOTES="GPU smoke test on synthetic data" \
run smoke data.source=synthetic train.dry_run=true logging.wandb=false

# A.1  Parity with the old trainer: same cell as imbalance_train.py's balanced
#      baseline (deepseek-tiny, aux 0.01, seed 42). Submit the old run next to it and
#      overlay train/lm_loss. Expect a close but not bit-identical curve (README:
#      token-1 embedding fix, T vs T-1 targets).
NOTES="parity vs imbalance_train aux0.01 seed42" \
run parity train.n_steps=300

# A.2  DDP: 4 GPUs at the same global batch (grad_accum 16 -> 4). Should track A.1.
GPUS=4 NOTES="DDP 4 GPUs at same tokens/step as parity" \
run parity-ddp4 train.n_steps=300 data.grad_accum=4 

# =====================================================================================
# B — baseline grid (2000 steps, deepseek-tiny): balancing strength x seed
# =====================================================================================

for seed in 0 1 2; do
    NOTES="balanced baseline" run "aux0.01-s$seed" loss.aux_coef=0.01 train.seed=$seed
    NOTES="weak balancing"    run "aux0.001-s$seed" loss.aux_coef=0.001 train.seed=$seed
    NOTES="no balancing (imbalanced regime)" run "aux0-s$seed" loss.aux_coef=0 train.seed=$seed
done

# B.1  Balanced baseline at the config's default seed (42) -- the same seed
#      imbalance_train.py's baseline cell used, so the two overlay directly.
#      RESUME=1 because ~4.5 h is long enough that a BACKOFF retry should continue
#      rather than restart from step 0.
RESUME=1 NOTES="balanced baseline, seed 42" \
run aux0.01-s42 loss.aux_coef=0.01 train.seed=42

# =====================================================================================
# C — other architectures / optimizer (seed 0)
# =====================================================================================

NOTES="olmoe-tiny top-8"     run olmoe-aux0.01 model.arch=olmoe
NOTES="qwen3moe-tiny"        run qwen3-aux0.01 model.arch=qwen3moe
# The old imb-adamw cell OOMed at the default batch; activation checkpointing should avoid that.
NOTES="AdamW control, imbalanced" \
run adamw-aux0 optim.name=adamw loss.aux_coef=0 parallel.activation_checkpointing=true

# C.4  deepseek at --scale regular (7.12B total / 1.28B active, 90% of params in the
#      experts) on ONE GPU. Weights + grads + Muon momentum + AdamW states are ~82 GiB
#      in fp32, so the micro-batch drops 16 -> 2 and activation checkpointing is on to
#      leave headroom on a 141 GB H200. grad_accum 16 -> 128 keeps tokens/step at the
#      same 524,288 as every other cell, so the loss curve stays comparable.
#      n_steps is 200, not 2000: ~8x the active params means ~8x the step time, and
#      2000 steps here would be ~2 days. If it still OOMs, halve data.batch_size to 1
#      and double data.grad_accum to 256.
TIME=24h RESUME=1 NOTES="deepseek regular (7B) scale check, 1 GPU" \
run deepseek-regular-trial model.scale=regular data.batch_size=2 data.grad_accum=128 \
    parallel.activation_checkpointing=true train.n_steps=200

# =====================================================================================
# D — longer runs (multi-GPU, resumable)
# =====================================================================================

GPUS=4 RESUME=1 TIME=48h NOTES="20k-step balanced baseline" \
run long-aux0.01 train.n_steps=20000 data.grad_accum=4

# =====================================================================================
# E — per-expert learning rate, imbalanced regime (aux 0), 2000 steps, seed 42
#     lr_e is proportional to s_e**alpha, renormalised to mean 1.0 over the experts
#     that got tokens, so alpha redistributes rate without moving the layer average.
#       alpha  1     token-proportional (restores SGD-like scaling)
#              0.5   square-root compromise -- the batch-1 winner in docs/RESULTS.md
#              0.25  weaker version of the same
#             -0.5   other sign: compensate cold experts upward (harmful in batch 1)
#     ema 0.9 smooths the counts over ~10 steps; ema 0 drives the rule off this
#     step's raw counts, which are very noisy exactly where n_e is small. Deadness
#     is keyed on raw counts either way, so that is not what the pair is testing.
#
#     alpha=0 is deliberately ONE cell, not two: `per_expert_lr.enabled` is
#     (alpha != 0 or freeze_dead), so at alpha=0 the callback is never constructed
#     and ema/normalize do nothing. Running it per-ema would submit the same
#     experiment twice under two names.
# =====================================================================================

# E.0  the control every E cell is read against: imbalanced, no intervention.
RESUME=1 NOTES="aux0 control, no per-expert LR (alpha=0 makes ema/normalize inert)" \
run pe-a0 loss.aux_coef=0 logging.expert_stats_every=50

for counts in raw ema; do
    case $counts in
        raw) ema=0 ;;       # rule driven by this step's counts
        ema) ema=0.9 ;;     # smoothed over ~10 steps
    esac
    for a in -0.5 0.25 0.5 1; do
        case $a in
            -*) lbl="aneg${a#-}" ;;   # -0.5 -> aneg0.5, keeps job names readable
             *) lbl="a$a" ;;
        esac
        RESUME=1 NOTES="per-expert LR alpha=$a on $counts counts, aux0" \
        run "pe-$counts-$lbl" loss.aux_coef=0 \
            optim.per_expert_lr.alpha=$a \
            optim.per_expert_lr.ema=$ema \
            optim.per_expert_lr.normalize=true \
            logging.expert_stats_every=50
    done
done

# -------------------------------------------------------------------------------------
[ "$n_matched" -eq 0 ] && {
    echo "No run named '$ONLY'. Names are the first argument to \`run\` below."; exit 1; }
echo "Done: $n_matched job(s). Logs: \$SCRATCH_HOME/$REPO_DIR/logs/ms-*-$BATCH.log"
echo "Status: runai list jobs | grep $BATCH"
