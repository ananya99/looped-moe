#!/usr/bin/env bash
# ===================================================================
# Per-expert learning rate experiment -> MLO RunAI cluster (via csub.py)
# ===================================================================
#
#   bash moestack/submit_expert_lr.sh                 # submit all 10 jobs
#   bash moestack/submit_expert_lr.sh baseline        # just this one (exact name)
#   bash moestack/submit_expert_lr.sh --dry | grep '>>'   # print, submit nothing
#   bash moestack/submit_expert_lr.sh --trial ema-a0.5    # 100 steps, wandb tag "trial"
#   bash moestack/submit_expert_lr.sh --batch b1019-1500 ema-a0.5   # reuse a tag (resume)
#
# Everything that defines the experiment is in the EDIT HERE block below -- arch,
# scale, GPU pool, steps, batch, the alpha and ema lists. Change it there, not on
# the command line, so the file is the record of what was run.
#
# The cells (all seed 42, 2000 steps, normalize on):
#
#   baseline           aux 0.01, no per-expert LR     balanced reference
#   a0                 aux 0,    no per-expert LR     imbalanced control
#   {raw,ema}-a{...}   aux 0,    alpha varied         the grid
#
# raw = per_expert_lr.ema 0, i.e. the rule reads this step's counts; ema = 0.9
# smooths over ~10 steps. Deadness is keyed on raw counts either way.
#
# alpha=0 is ONE cell, not two: per_expert_lr.enabled is (alpha != 0 or
# freeze_dead), so at alpha=0 the callback is never constructed and ema does
# nothing. Submitting it per-ema would run the same experiment twice.
#
# Compare cells on train/lm_loss, never train/loss -- baseline has aux 0.01 and
# the rest have 0, so only the pure cross-entropy is comparable.
#
# Each job: runs from the cluster checkout (push AND pull there first -- there is
# no branch guard; the first log line echoes the commit), tees to
# $SCRATCH_HOME/looped-moe/logs/<job>.log, and is capped by `timeout` because
# csub drops -t when --command is given. On expiry it exits 124 and runai retries
# up to $BACKOFF times, so only RESUME=1 runs make progress across retries.
# ===================================================================

set -euo pipefail

# ===================================================================
# EDIT HERE
# ===================================================================
CSUB=/Users/ananyagupta/repos/getting-started/csub.py   # cluster wrapper, on this laptop
REPO_DIR=looped-moe                 # relative to the job's working dir (scratch home)
VENV=.venv                          # relative to REPO_DIR

ARCH=deepseek                       # deepseek | olmoe | qwen3moe
SCALE=regular                       # tiny | small | regular
OPTIMIZER=muon                      # per-expert LR requires muon (it raises under adamw)
NODE_TYPE=h200
GPUS=1                              # >1 switches the launcher to torchrun

N_STEPS=2000
SEED=42
LR=0.02

# bs x ga = 256 holds tokens/step at 524,288 for every scale, so a cell always sees
# the same 1.05B tokens and stays inside the cache's single epoch. Measured on one
# H200 (150.8 GB) at regular, 7.12B params / 872M active per token:
#
#   2x128  94.0 GB  12.7k tok/s      checkpointed activations cost ~1.2 GB/sample
#   4x64   97.1 GB  14.4k tok/s      on a ~91 GB floor of params + grads + Muon
#   8x32  101.7 GB  15.6k tok/s      momentum + AdamW moments. Bigger micro-batches
#  16x16  113.3 GB  16.4k tok/s  <-  are strictly faster, so take the largest that
#                                    still leaves room (37 GB spare) -> ~18 h/cell.
#
# ACT_CKPT is not tuning at regular scale: unchecked activations cost ~12.7 GB per
# sample, so bs=4 alone peaks at 143.4 GB and bs=16 cannot run at all.
# small is untested -- 8x32 is a conservative carry-over and can probably take 16x16.
case "$SCALE" in
    tiny)    BATCH_SIZE=16; GRAD_ACCUM=16; ACT_CKPT=false ;;
    small)   BATCH_SIZE=8;  GRAD_ACCUM=32; ACT_CKPT=true  ;;
    regular) BATCH_SIZE=16; GRAD_ACCUM=16; ACT_CKPT=true  ;;
esac

AUX_BASELINE=0.01                   # the balanced reference cell's aux coefficient
STATS_EVERY=50                      # per-expert JSONL dump cadence; 0 disables
SAVE_EVERY=250                      # a reclaim costs at most this much work:
                                    # 2.2 h at regular, 69 min at small. Each
                                    # regular checkpoint is ~60 GB and they are
                                    # kept, so this is ~480 GB per cell.

ALPHAS="-0.5 0.25 0.5 1"            # 0 is the control cell below, not part of the sweep
EMAS="0"                        # 0 = raw per-step counts

TIME=24h                            # per attempt; regular is ~18 h/cell, so one
                                    # attempt covers a cell with margin
BACKOFF=4                           # runai retries before the job is marked Failed
# ===================================================================

JOB_PREFIX=pe                       # k8s job names: <prefix>-<name>-<batch>
CACHE_SEQUENCES=566129              # what conf/baseline.yaml's disk cache holds

DRY="${DRY:-0}"
TRIAL=0
TRIAL_STEPS=100
BATCH="${BATCH:-b$(date +%m%d-%H%M)}"

while [ $# -gt 0 ]; do
    case "$1" in
        --dry|-n)  DRY=1;   shift ;;
        --trial)   TRIAL=1; shift ;;
        --batch)   shift; BATCH="${1:-}"; shift || true
                   [ -n "$BATCH" ] || { echo "--batch needs a tag, e.g. --batch b1019-1500" >&2; exit 1; } ;;
        --batch=*) BATCH="${1#--batch=}"; shift
                   [ -n "$BATCH" ] || { echo "--batch needs a tag" >&2; exit 1; } ;;
        -h|--help) sed -n '2,37p' "$0"; exit 0 ;;
        -*)        echo "unknown flag: $1 (try --help)" >&2; exit 1 ;;
        *)         break ;;
    esac
done
ONLY="${1:-}"

# Applied to every cell. These equal the conf/baseline.yaml defaults as shipped, so
# they do not move the run-name fingerprint -- they are here so the file states the
# whole experiment rather than half of it.
COMMON=(
    "model.arch=$ARCH" "model.scale=$SCALE" "optim.name=$OPTIMIZER" "optim.lr=$LR"
    "data.batch_size=$BATCH_SIZE" "data.grad_accum=$GRAD_ACCUM"
    "train.n_steps=$N_STEPS" "train.seed=$SEED"
    "logging.expert_stats_every=$STATS_EVERY" "train.save_every=$SAVE_EVERY"
    "parallel.activation_checkpointing=$ACT_CKPT"
)

# The disk cache wraps silently, so say so before 10 jobs quietly train two epochs.
SEQUENCES=$(( BATCH_SIZE * GRAD_ACCUM * N_STEPS * GPUS ))
if [ "$SEQUENCES" -gt "$CACHE_SEQUENCES" ]; then
    echo "WARNING: each cell needs $SEQUENCES sequences but the cache holds $CACHE_SEQUENCES."
    echo "         Runs will wrap and repeat data. Rebuild with a larger --tokens," \
         "or lower N_STEPS." >&2
fi

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

run () {   # [NOTES=..] [RESUME=1] run <name> <extra hydra overrides...>
    local base="$1"; shift
    local notes="${NOTES:-}" resume="${RESUME:-1}"
    local secs; secs="$(to_seconds "$TIME")"   # validated before anything is submitted
    local job
    job="$(echo "$JOB_PREFIX-$base-$BATCH" | tr '[:upper:]' '[:lower:]' \
           | tr '_.' '--' | tr -cd 'a-z0-9-' | cut -c1-63)"

    # ${arr[@]+"${arr[@]}"}: macOS bash 3.2 calls an empty array unbound under set -u
    local overrides=(${COMMON[@]+"${COMMON[@]}"} ${1+"$@"} "logging.tag=$BATCH")
    [ "$resume" = 1 ] && overrides+=("train.auto_resume=true")
    [ -n "$notes" ] && overrides+=("logging.notes='$notes'")
    # Last, so Hydra's last-value-wins beats the N_STEPS in COMMON.
    [ "$TRIAL" = 1 ] && overrides+=("train.n_steps=$TRIAL_STEPS" "logging.tags=[trial]")

    if [ -n "$ONLY" ] && [ "$ONLY" != "$base" ]; then
        return
    fi
    n_matched=$((n_matched + 1))

    local launcher="python -m moestack.train"
    [ "$GPUS" -gt 1 ] && launcher="torchrun --standalone --nproc_per_node=$GPUS -m moestack.train"

    local args="" o
    for o in "${overrides[@]}"; do args+=" $(printf %q "$o")"; done

    local cmd="set -o pipefail && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
cd $REPO_DIR && source $VENV/bin/activate && mkdir -p logs && \
echo \"[$job] commit \$(git rev-parse --short HEAD)\$(git diff --quiet || echo ' (dirty)')\" | tee -a logs/$job.log && \
timeout -k 120 $secs $launcher$args 2>&1 | tee -a logs/$job.log; rc=\$?; \
[ \$rc -eq 124 ] && echo \"[$job] stopped: $TIME wall-clock budget exhausted\" | tee -a logs/$job.log; \
exit \$rc"

    # no -t: csub only uses --time for its keep-alive sleep, which --command replaces.
    local flags=(-n "$job" -g "$GPUS" --node-type "$NODE_TYPE" --train --backofflimit "$BACKOFF")
    [ "$GPUS" -gt 1 ] && flags+=(--large-shm)
    [ "$DRY" = 1 ] && flags+=(--dry)

    echo ">> $job"
    python3 "$CSUB" "${flags[@]}" --command "$cmd"
}

trial_note=""
[ "$TRIAL" = 1 ] && trial_note="  trial=$TRIAL_STEPS steps"
echo "Submitting batch $BATCH  ${ONLY:+only='$ONLY'  }dry=$DRY$trial_note"
echo "  $ARCH-$SCALE $OPTIMIZER lr=$LR  ${BATCH_SIZE}x${GRAD_ACCUM}x2048 x ${N_STEPS} steps" \
     "on ${GPUS}x$NODE_TYPE  ($SEQUENCES sequences/cell)"

# -------------------------------------------------------------------------------------
# 1. References
# -------------------------------------------------------------------------------------

# Balanced: the load-balancing loss does the work. What the grid has to beat.
NOTES="balanced reference, aux $AUX_BASELINE" \
run baseline "loss.aux_coef=$AUX_BASELINE"

# Imbalanced, untouched. alpha=0 leaves per_expert_lr disabled entirely, so this is
# the plain baseline in the regime the grid intervenes in.
NOTES="imbalanced control, no per-expert LR" \
run a0 "loss.aux_coef=0"

# -------------------------------------------------------------------------------------
# 2. The grid: {raw, ema} x alpha, all at aux 0 with normalize on
# -------------------------------------------------------------------------------------

for ema in $EMAS; do
    case "$ema" in
        0|0.0) counts=raw ;;
        *)     counts=ema ;;
    esac
    for a in $ALPHAS; do
        case "$a" in
            -*) lbl="aneg${a#-}" ;;   # -0.5 -> aneg0.5, so job names stay readable
             *) lbl="a$a" ;;
        esac
        NOTES="alpha=$a on $counts counts, aux 0" \
        run "$counts-$lbl" \
            "loss.aux_coef=0" \
            "optim.per_expert_lr.alpha=$a" \
            "optim.per_expert_lr.ema=$ema" \
            "optim.per_expert_lr.normalize=true"
    done
done

# -------------------------------------------------------------------------------------
[ "$n_matched" -eq 0 ] && {
    echo "No run named '$ONLY'. Names: baseline, a0, raw-a*, ema-a* (see --dry)."; exit 1; }
echo "Done: $n_matched job(s). Logs: \$SCRATCH_HOME/$REPO_DIR/logs/$JOB_PREFIX-*-$BATCH.log"
echo "Status: runai list jobs | grep $BATCH"
