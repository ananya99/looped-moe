"""Run identity and experiment tracking for imbalance_train.py.

Everything that names, groups and labels a run lives here, so the training
script only contains the experiment itself:

  build_run_name        unique run name (checkpoints, JSONLs, wandb name)
  build_condition_name  run name minus --tag: one value per experimental condition,
                        used as the wandb group so seeds/batches of a condition
                        group together
  default_job_type      baseline / intervention / control, inferred from flags
  git_state             (commit, dirty) of the running code
  add_tracking_args     the wandb CLI flags
  init_wandb            assembles group, tags, notes and config, then wandb.init
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def git_state():
    """(commit, dirty) of the code this run executes, or (None, None) outside git.

    Logged to wandb so a run made with stale code on the cluster is visible
    from the config alone.
    """
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                             capture_output=True, text=True, timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=here, capture_output=True, text=True,
                                    timeout=10).stdout.strip())
        return sha, dirty
    except Exception:
        return None, None


def default_job_type(args):
    """Role of the run in the experiment, for colouring/filtering in wandb."""
    if args.optimizer != "muon":
        return "control"
    if args.per_expert_lr_alpha != 0.0 or args.freeze_dead_experts:
        return "intervention"
    return "baseline"


def build_condition_name(args):
    """The run name without the batch tag: one value per experimental condition.

    Used as the wandb group, so repeated runs of a condition (other seeds, other
    batches) land in one group and wandb can plot their mean and spread.
    """
    tag = args.tag
    args.tag = None
    try:
        return build_run_name(args)
    finally:
        args.tag = tag


def build_run_name(args):
    """Encode every flag that changes the experiment, so runs cannot collide.

    moe_train.py derives its run name from architecture and tying only, which
    means two runs differing solely in a learning-rate flag overwrite each
    other's checkpoints. Here every knob that defines a matrix cell is in the name.
    """
    def fmt(x):
        return f"{x:g}"

    parts = [args.arch]
    if args.scale != "regular":
        parts.append(args.scale)
    parts.append(f"aux{fmt(args.aux_loss_coef)}")
    if args.per_expert_lr_alpha != 0.0:
        a = args.per_expert_lr_alpha
        # "aneg0.5" rather than "a-0.5": a bare minus reads as a separator in
        # both the filename and the wandb run name.
        parts.append(f"aneg{fmt(abs(a))}" if a < 0 else f"a{fmt(a)}")
    if args.freeze_dead_experts:
        parts.append("freeze")
    if args.optimizer != "muon":
        parts.append(args.optimizer)
    if args.tag:
        parts.append(args.tag)
    return "-".join(parts)


# ──────────────────────────────────────────────────────────────────────
# 6. Training
# ──────────────────────────────────────────────────────────────────────


def add_tracking_args(p):
    """Register the wandb-related CLI flags on an argparse parser."""
    p.add_argument("--wandb-project", default="moe-expert-imbalance")
    p.add_argument("--run-group", default=None,
                   help="wandb group; default = the condition name (run name minus "
                        "--tag), so seeds/batches of one condition group together")
    p.add_argument("--wandb-tags", default=None,
                   help="Comma-separated extra wandb tags; batch:<tag> and "
                        "steps:<n> are added automatically")
    p.add_argument("--wandb-notes", default=None,
                   help="One-line purpose of the run, shown in wandb")
    p.add_argument("--wandb-job-type", default=None,
                   help="Default: baseline / intervention / control, inferred from flags")


def init_wandb(args, run_name: str, n_steps: int, config: dict, resume_id=None):
    """Start (or resume) the wandb run with full identity metadata.

    ``config`` holds the experiment-specific values (model sizes, effective batch,
    ...). On top of it this logs every CLI flag, the effective step count, the
    condition name and the git state, and sets group, job_type, tags and notes.
    """
    import wandb

    condition = build_condition_name(args)
    git_sha, git_dirty = git_state()

    tags = [t.strip() for t in (args.wandb_tags or "").split(",") if t.strip()]
    if args.tag:
        tags.append(f"batch:{args.tag}")
    tags.append(f"steps:{n_steps}")
    if git_dirty:
        tags.append("git-dirty")

    kwargs = {
        "project": args.wandb_project,
        "group": args.run_group or condition,
        "job_type": args.wandb_job_type or default_job_type(args),
        "tags": sorted(set(tags)),
        "notes": args.wandb_notes,
        "config": {
            # Every CLI flag first; experiment values and identity override it.
            **vars(args),
            **config,
            "n_steps": n_steps,
            "seed": args.seed,
            "tag": args.tag,
            "run_name": run_name,
            "condition": condition,
            "git_commit": git_sha,
            "git_dirty": git_dirty,
        },
    }
    if resume_id:
        kwargs.update(id=resume_id, resume="must")
    else:
        kwargs["name"] = run_name
    return wandb.init(**kwargs)
