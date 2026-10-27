#!/usr/bin/env bash
# ===================================================================
# Expert-local gradient accumulation -> MLO RunAI cluster (via csub.py)
# ===================================================================
#
#   bash moestack/submit_expert_accum.sh                  # submit all 10 jobs
#   bash moestack/submit_expert_accum.sh st-r4            # just this one (exact name)
#   bash moestack/submit_expert_accum.sh --dry | grep '>>'    # print, submit nothing
#   bash moestack/submit_expert_accum.sh --trial mb-r4    # 100 steps, wandb tag "trial"
#
# The question: the baseline gives every expert one update per step, built from
# whatever the router happened to send it -- a big-batch estimate for a hot
# expert, a handful of tokens for a cold one, renormalised stale momentum for a
# dead one. Fix the token budget per update instead and let the update *rate*
# vary. Does an expert learn better from one update per Q tokens than from a
# fixed number of updates of wildly varying quality?
#
# The cells (all seed 42, aux 0 except the balanced reference):
#
#   baseline     aux 0.01, no accumulation      balanced reference
#   a0           aux 0,    no accumulation      the imbalanced regime, untouched
#   active       Q=1, D=1                       update only the experts that got
#                                               tokens -- separates "stop stepping
#                                               dead experts" from accumulation
#   st-r{2,4,8}  step clock, Q = r x a balanced expert's per-step load
#   mb-r{2,4,8}  micro-batch clock, the SAME absolute Q
#   st-r4-sum    step clock, grad_scale=sum     is dividing by the window's tokens
#                                               doing anything?
#
# Two conversions the grid depends on, both easy to get wrong:
#
#   Q  threshold_rel counts multiples of a balanced expert's load *per tick*, and
#      a tick is a micro-batch on the mb cells. So the mb cells pass an ABSOLUTE
#      threshold equal to the st cells' Q; threshold_rel=4 on both clocks would
#      be two different experiments.
#   D  max_window is in ticks too, so the mb cells' staleness cap is D_STEP times
#      grad_accum -- otherwise "16" means 16 steps on one clock and one step on
#      the other.
#
# AUTO-RESUME IS ON, and a cell needs it: the H200 pool preempts, and the first
# batch of this grid lost 72 of 76 attempts that way (median 1.8 h, against the
# ~18 h a cell takes). Pods are deleted and recreated rather than restarted, so
# --backofflimit does not cap it and SIGTERM never reaches the trainer -- the
# 30 s grace period is far too short to reach a step boundary and write a 59 GB
# checkpoint, so periodic saves are the only thing that survives.
#
# What a resume costs: the open windows live in the gradient buffers and are NOT
# checkpointed, so they are dropped on load (`accum/resumes` and
# `accum/resume_dropped_windows` say how often and how many). The event history
# carries over. A cell with many resumes has lost a little accumulated gradient
# that one with none has not -- check those two series before comparing cells.
#
# Hence SAVE_EVERY is low and keep_last_n is set: a save costs ~5 min of the
# ~53 min it now buys back, and without retention 20 saves x 59 GB would be
# 1.2 TB per cell.
#
# Compare cells on train/lm_loss, never train/loss -- the baseline has aux 0.01
# and the rest have 0, so only the pure cross-entropy is comparable.
#
# Watch accum/event_gini next to load/gini. The scheme can feed a loop: a hot
# expert crosses its budget more often, takes more steps, improves, and the
# router sends it still more. If event_gini runs away while load/gini does not,
# the extra steps are doing it rather than the routing.
#
# Each job: runs from the cluster checkout (push AND pull there first -- there is
# no branch guard; the first log line echoes the commit), tees to
# $SCRATCH_HOME/looped-moe/logs/<job>.log, and is capped by `timeout` because
# csub drops -t when --command is given.
# ===================================================================

set -euo pipefail

# ===================================================================
# EDIT HERE
# ===================================================================
CSUB=/Users/ananyagupta/repos/getting-started/csub.py   # cluster wrapper, on this laptop
# Overridable from the environment because they are about *where* a run happens,
# not what it is:  REPO_DIR=looped-moe bash moestack/submit_expert_accum.sh sends a
# cell to the main checkout instead.
#
# Defaults to the accum worktree, NOT the main checkout: `looped-moe` sits on
# refactor/baseline for the per-expert-LR runs, and none of optim.expert_accum
# exists on that branch -- a cell submitted there dies on the first Hydra override.
REPO_DIR="${REPO_DIR:-looped-moe-accum}"  # relative to the job's working dir (scratch home)
VENV="${VENV:-.venv}"               # relative to REPO_DIR

ARCH=deepseek                       # deepseek | olmoe | qwen3moe
# Edit here, not from the command line. The grid is `regular`; drop to `tiny` for
# a smoke test on a GPU and put it back afterwards. The run name carries the scale
# either way, so a trial can never be mistaken for a cell.
SCALE=tiny                          # tiny | small | regular
OPTIMIZER=muon                      # accumulation requires muon (it raises under adamw)
NODE_TYPE=h200
GPUS=1                              # the mb cells are single-process only

N_STEPS=2000
SEED=42
LR=0.02
SEQ_LEN=2048

# Routing shape, needed to turn threshold_rel into the absolute Q the mb cells
# take. deepseek/olmoe are 64 experts (top-6 / top-8); qwen3moe is 60 top-4.
case "$ARCH" in
    olmoe)    TOP_K=8; N_EXPERTS=64 ;;
    deepseek) TOP_K=6; N_EXPERTS=64 ;;
    qwen3moe) TOP_K=4; N_EXPERTS=60 ;;
esac

# Same geometry as submit_expert_lr.sh: bs x ga = 256 holds tokens/step at
# 524,288 for every scale, so a cell sees the same 1.05B tokens.
case "$SCALE" in
    tiny)    BATCH_SIZE=16; GRAD_ACCUM=16; ACT_CKPT=false ;;
    small)   BATCH_SIZE=8;  GRAD_ACCUM=32; ACT_CKPT=true  ;;
    regular) BATCH_SIZE=16; GRAD_ACCUM=16; ACT_CKPT=true  ;;
esac

AUX_BASELINE=0.01                   # the balanced reference cell's aux coefficient
STATS_EVERY=50                      # per-expert drift JSONL cadence; 0 disables
EVENTS_EVERY=50                     # accumulation events JSONL cadence; 0 disables
# ~53 min of training between saves, so a preemption costs ~26 min on average
# instead of the whole cell. Each save is ~5 min, so this is ~9% overhead -- the
# price of the pool being preemptible.
SAVE_EVERY=100
KEEP_LAST=2                         # 59 GB a file; keep the resume point and one spare

RELS="1 1.5 2"                        # Q, in multiples of a balanced expert's load
D_STEP=16                           # staleness cap on the step clock, in steps

TIME=30h                            # per attempt; auto-resume carries a cell across several
BACKOFF=1                           # inert here: evicted pods are recreated, not retried
# ===================================================================

JOB_PREFIX=ea                       # k8s job names: <prefix>-<name>-<batch>
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
    "train.keep_last_n=$KEEP_LAST"
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
    # Resume by default now that accumulation runs support it: RESUME=0 opts out.
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

# A balanced expert's tokens per step -- what threshold_rel multiplies on the step
# clock. Computed here so the mb cells can be handed the same Q in absolute tokens
# rather than a per-micro-batch multiple of it.
Q_STEP=$(( BATCH_SIZE * GRAD_ACCUM * SEQ_LEN * GPUS * TOP_K / N_EXPERTS ))
D_MB=$(( D_STEP * GRAD_ACCUM ))     # the same staleness, counted in the other tick

# RELS may hold fractions (1.5x a balanced load), which $(( )) cannot multiply, and
# `threshold` is an integer token count -- so round through awk.
q_abs () {   # q_abs <rel> -> Q_STEP x rel in whole tokens
    awk -v q="$Q_STEP" -v r="$1" 'BEGIN { printf "%d", q * r + 0.5 }'
}

echo "Submitting batch $BATCH  ${ONLY:+only='$ONLY'  }dry=$DRY$trial_note"
echo "  $ARCH-$SCALE $OPTIMIZER lr=$LR  ${BATCH_SIZE}x${GRAD_ACCUM}x${SEQ_LEN} x ${N_STEPS} steps" \
     "on ${GPUS}x$NODE_TYPE  ($SEQUENCES sequences/cell)"
echo "  balanced load $Q_STEP tokens/expert/step; D=$D_STEP steps = $D_MB micro-batches"

ACCUM=( "logging.accum_events_every=$EVENTS_EVERY" "optim.expert_accum.enabled=true" )

# -------------------------------------------------------------------------------------
# 1. References -- no accumulation at all
# -------------------------------------------------------------------------------------

# Balanced: the load-balancing loss does the work. What the grid has to beat.
NOTES="balanced reference, aux $AUX_BASELINE, no accumulation" \
# run baseline "loss.aux_coef=$AUX_BASELINE"

# Imbalanced and untouched: every expert updates every step, including the ones
# that saw nothing. This is the regime the grid intervenes in.
NOTES="imbalanced control, no accumulation" \
run a0 "loss.aux_coef=0"

# -------------------------------------------------------------------------------------
# 2. The control that is not accumulation: just stop stepping dead experts
# -------------------------------------------------------------------------------------
#
# Q=1 with D=1 closes a window every tick, so nothing accumulates anywhere -- the
# only difference from `a0` is that a zero-token expert takes no step at all. Any
# gain the real cells show has to be measured against THIS, not against a0, or it
# is impossible to say whether the budget did the work or simply not stepping on
# renormalised stale momentum did.
NOTES="active-only control: no accumulation, but no stepping zero-token experts" \
run active "loss.aux_coef=0" "${ACCUM[@]}" \
    "optim.expert_accum.clock=step" \
    "optim.expert_accum.threshold=1" "optim.expert_accum.max_window=1"

# -------------------------------------------------------------------------------------
# 3. The step clock: baseline cadence, but only when the budget is met
# -------------------------------------------------------------------------------------

for r in $RELS; do
    NOTES="step clock, Q=${r}x balanced per-step load, D=$D_STEP steps, aux 0" \
    run "st-r$r" "loss.aux_coef=0" "${ACCUM[@]}" \
        "optim.expert_accum.clock=step" \
        "optim.expert_accum.threshold_rel=$r" \
        "optim.expert_accum.max_window=$D_STEP"
done

# -------------------------------------------------------------------------------------
# 4. The micro-batch clock: the same budgets, but experts stop waiting for the batch
# -------------------------------------------------------------------------------------
#
# Absolute Q and a converted D, so these differ from the st cells in exactly one
# thing: when an expert may act on a full window, not how full it is.
for r in $RELS; do
    q="$(q_abs "$r")"
    NOTES="micro-batch clock, Q=$q tokens (${r}x per-step load), D=$D_MB ticks, aux 0" \
    run "mb-r$r" "loss.aux_coef=0" "${ACCUM[@]}" \
        "optim.expert_accum.clock=microbatch" \
        "optim.expert_accum.threshold=$q" \
        "optim.expert_accum.max_window=$D_MB"
done

# -------------------------------------------------------------------------------------
# 5. Does dividing by the window's tokens matter?
# -------------------------------------------------------------------------------------
#
# Under Muon the overall gradient scale washes out in Newton-Schulz, so this is
# not a step-size question: it is about `buf.lerp_(grad, 1 - mom)`, which weighs
# an event into the momentum buffer in proportion to its magnitude. With `sum`, a
# window spanning eight ticks dominates one spanning two purely by being longer.
NOTES="step clock, Q=4x, raw summed gradient instead of the per-token mean" \
run st-r4-sum "loss.aux_coef=0" "${ACCUM[@]}" \
    "optim.expert_accum.clock=step" \
    "optim.expert_accum.threshold_rel=4" \
    "optim.expert_accum.max_window=$D_STEP" \
    "optim.expert_accum.grad_scale=sum"

# -------------------------------------------------------------------------------------
[ "$n_matched" -eq 0 ] && {
    echo "No run named '$ONLY'. Names: baseline, a0, active, st-r*, mb-r*, st-r4-sum (see --dry)."; exit 1; }
echo "Done: $n_matched job(s). Logs: \$SCRATCH_HOME/$REPO_DIR/logs/$JOB_PREFIX-*-$BATCH.log"
echo "Status: runai list jobs | grep $BATCH"
