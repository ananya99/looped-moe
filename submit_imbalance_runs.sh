#!/bin/bash
# ===================================================================
# Expert-load imbalance warm-up experiment
# 8 runs x ~1h on a single H200 (tiny scale, 2000 steps)
# ===================================================================
#
# Question (Martin, Slack): how does the number of tokens per expert influence
# the learning of those expert weights and router weights, and can we vary the
# learning rate per expert based on it? Imbalance is largest early, so short
# runs are the right instrument.
#
# Runs against imbalance_train.py — moe_train.py with the expert-tying machinery
# removed, which is inert at g=1 anyway. Every expert is its own Muon parameter
# group there, so per-expert learning rates are always addressable.
#
# Sections:
#   A — measurement only (alpha = 0). Establishes what the imbalance actually is
#       and whether token share predicts how much an expert moves.
#   B — interventions, all in the imbalanced regime (--aux-loss-coef 0).
#   C — controls.
#
# Read on train/lm_loss (pure cross-entropy, aux/z removed), NOT train/loss:
# the runs differ in their aux coefficient, so train/loss is not comparable
# across sections. Per-expert records land in
# checkpoints/<run_name>_expert_stats.jsonl, where run_name encodes every flag
# that defines the cell, so runs cannot overwrite each other.
#
# --auto-resume is deliberately off: on a 1h run, replaying the stream
# micro-batch by micro-batch costs more than restarting from scratch.
# ===================================================================

# unused flags: --no-gradient-checkpointing 
COMMON="--arch deepseek --scale tiny --optimizer muon --z-loss-coef 1e-4 \
--batch-size 16 --grad-accum 16 --n-steps 2000 --no-gradient-checkpointing \
--freeze-dead-experts"
# wandb group defaults to the condition (run name minus the batch tag), so seeds
# and batches of the same condition group together. Don't pass --run-group here.

csub_path="/Users/ananyagupta/repos/getting-started/csub.py"

# One tag per submission, computed once so every job in this batch shares it.
# It goes into the cluster job name, the log file, and (via --tag) the run name,
# so checkpoints, JSONLs, wandb names and logs never collide with earlier batches.
# Override to reuse a tag, e.g.  BATCH=b1006-0930 bash submit_imbalance_runs.sh
BATCH="${BATCH:-b$(date +%m%d-%H%M)}"
echo "Submitting batch $BATCH"

submit () {   # [NOTES="purpose"] submit <job-name> <extra flags...>
    local name="$1-$BATCH"; shift
    # %q-quote the notes so spaces/quotes survive the remote shell.
    local notes_flag=""
    [ -n "$NOTES" ] && notes_flag="--wandb-notes $(printf %q "$NOTES")"
    python3 "$csub_path" -n "$name" -g 1 --node-type h200 --train -t 16h \
        --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
source looped-moe/.venv/bin/activate && cd looped-moe && mkdir -p logs && \
python imbalance_train.py $COMMON --tag $BATCH $notes_flag $* 2>&1 | tee -a logs/$name.log"
}

# ===================================================================
# SECTION A — measurement
# ===================================================================

# A.0  Balanced reference: load balancing at its default strength.
submit imb-bal-a0 --aux-loss-coef 0.01

# A.1  THE measurement run: load balancing off, so the imbalance is real.
submit imb-none-a0 --aux-loss-coef 0

# A.2  Intermediate strength, to check the effect is monotone in the coefficient.
# submit imb-mid-a0 --aux-loss-coef 0.001

# ===================================================================
# SECTION B — per-expert learning-rate interventions (imbalanced regime)
# ===================================================================

# B.0  Freeze zero-token experts. Run this one first: a dead expert otherwise
#      takes a full-magnitude Muon step from a stale momentum buffer and is
#      shrunk by weight decay every step, and that alone may account for most
#      of whatever the smarter rules buy.
# submit imb-freeze --aux-loss-coef 0 --freeze-dead-experts

# B.1  lr_e proportional to sqrt(token share).
submit imb-a05 --aux-loss-coef 0 --per-expert-lr-alpha 0.5

# B.2  Fully token-proportional. Roughly restores the scaling that
#      Newton-Schulz normalisation removes.
# submit imb-a1 --aux-loss-coef 0 --per-expert-lr-alpha 1.0

# B.3  Opposite sign: compensate cold experts upward. The correct sign is not
#      obvious a priori, which is the point of running it.
# submit imb-aneg05 --aux-loss-coef 0 --per-expert-lr-alpha -0.5

# ===================================================================
# SECTION C — controls
# ===================================================================

# C.0  Does the picture survive a different normaliser? AdamW keeps the expert
#      weights as single 3D tensors, so per-expert LR is not available there —
#      this run is measurement-only by construction.
# submit imb-adamw --aux-loss-coef 0 --optimizer adamw
