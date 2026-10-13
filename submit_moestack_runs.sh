#!/usr/bin/env bash
# ===================================================================
# Submit moestack runs to the MLO RunAI cluster (via csub.py)
# ===================================================================
#
#   bash submit_moestack_runs.sh                 # list the runs defined below, submit nothing
#   bash submit_moestack_runs.sh all             # submit every run
#   bash submit_moestack_runs.sh 'parity|aux0'   # submit runs whose name matches the regex
#   DRY=1 bash submit_moestack_runs.sh all       # print the runai commands (csub --dry)
#   BATCH=b1013-0930 bash ... aux0               # reuse a batch tag (e.g. to resume)
#
# Each run is one line:  [GPUS=n] [NOTES="..."] [RESUME=1] run <name> <hydra overrides...>
#   GPUS     1 -> python -m moestack.train;  n>1 -> torchrun --nproc_per_node=n (DDP)
#   NOTES    one-line purpose, goes to wandb notes
#   RESUME   1 -> train.auto_resume=true (worth it for long runs; short runs restart faster)
#   overrides are Hydra key=value pairs over moestack/conf/baseline.yaml,
#   e.g. loss.aux_coef=0 train.seed=1 data.grad_accum=4
#
# Every job:
#   - runs from $REPO_DIR in your scratch home and refuses to start unless that
#     checkout is on $EXPECT_BRANCH (so a stale main checkout can't silently run);
#   - logs the git commit, then tees stdout/stderr to $REPO_DIR/logs/<job>.log;
#   - gets logging.tag=$BATCH, so run names, checkpoints, wandb names and logs of
#     different submissions never collide (wandb group = run name minus the tag).
#
# Compare runs on train/lm_loss, not train/loss, whenever aux/z coefficients differ.
# ===================================================================

set -euo pipefail

# ---- cluster / environment (override any of these from the environment) -------
CSUB="${CSUB:-/Users/ananyagupta/repos/getting-started/csub.py}"
REPO_DIR="${REPO_DIR:-looped-moe}"          # relative to the job's working dir (scratch home)
VENV="${VENV:-.venv}"                        # relative to REPO_DIR
EXPECT_BRANCH="${EXPECT_BRANCH:-refactor/baseline}"
NODE_TYPE="${NODE_TYPE:-h200}"
TIME="${TIME:-16h}"
DRY="${DRY:-0}"
BATCH="${BATCH:-b$(date +%m%d-%H%M)}"

FILTER="${1:-}"
# Hydra overrides applied to every run in this file.
COMMON=()

# ---- machinery -------------------------------------------------------------------
n_matched=0

run () {   # [GPUS=n] [NOTES=..] [RESUME=1] run <name> <overrides...>
    local base="$1"; shift
    local gpus="${GPUS:-1}" notes="${NOTES:-}" resume="${RESUME:-0}"
    # k8s job names: lowercase alphanumerics and '-', <= 63 chars
    local job
    job="$(echo "ms-$base-$BATCH" | tr '[:upper:]_.' '[:lower:]--' | tr -cd 'a-z0-9-' | cut -c1-63)"

    local overrides=("${COMMON[@]}" "$@" "logging.tag=$BATCH")
    [ "$resume" = 1 ] && overrides+=("train.auto_resume=true")
    [ -n "$notes" ] && overrides+=("logging.notes='$notes'")

    if [ -z "$FILTER" ]; then
        printf '  %-22s gpus=%-2s %s\n' "$base" "$gpus" "$*"
        return
    fi
    if [ "$FILTER" != all ] && ! [[ "$base" =~ $FILTER ]]; then
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
branch=\$(git rev-parse --abbrev-ref HEAD) && \
{ [ \"\$branch\" = \"$EXPECT_BRANCH\" ] || { echo \"[$job] checkout is on \$branch, expected $EXPECT_BRANCH\" | tee -a logs/$job.log; exit 1; }; } && \
echo \"[$job] commit \$(git rev-parse --short HEAD)\$(git diff --quiet || echo ' (dirty)')\" | tee -a logs/$job.log && \
$launcher$args 2>&1 | tee -a logs/$job.log"

    local flags=(-n "$job" -g "$gpus" --node-type "$NODE_TYPE" --train -t "$TIME")
    [ "$gpus" -gt 1 ] && flags+=(--large-shm)
    [ "$DRY" = 1 ] && flags+=(--dry)

    echo ">> $job  ($gpus GPU)"
    python3 "$CSUB" "${flags[@]}" --command "$cmd"
}

if [ -z "$FILTER" ]; then
    echo "Runs defined in $(basename "$0") (batch tag would be $BATCH):"
else
    echo "Submitting batch $BATCH  filter='$FILTER'  dry=$DRY  branch=$EXPECT_BRANCH"
fi

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

# =====================================================================================
# C — other architectures / optimizer (seed 0)
# =====================================================================================

NOTES="olmoe-tiny top-8"     run olmoe-aux0.01 model.arch=olmoe
NOTES="qwen3moe-tiny"        run qwen3-aux0.01 model.arch=qwen3moe
# The old imb-adamw cell OOMed at the default batch; activation checkpointing should avoid that.
NOTES="AdamW control, imbalanced" \
run adamw-aux0 optim.name=adamw loss.aux_coef=0 parallel.activation_checkpointing=true

# =====================================================================================
# D — longer runs (multi-GPU, resumable)
# =====================================================================================

GPUS=4 RESUME=1 TIME=48h NOTES="20k-step balanced baseline" \
run long-aux0.01 train.n_steps=20000 data.grad_accum=4

# -------------------------------------------------------------------------------------
if [ -n "$FILTER" ]; then
    [ "$n_matched" -eq 0 ] && { echo "No run matches '$FILTER'."; exit 1; }
    echo "Done: $n_matched job(s). Logs: \$SCRATCH_HOME/$REPO_DIR/logs/ms-*-$BATCH.log"
    echo "Status: runai list jobs | grep $BATCH"
else
    echo
    echo "Submit with:  bash $(basename "$0") all   |   bash $(basename "$0") '<regex>'"
fi
