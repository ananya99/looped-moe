"""CPU tests for expert_stats.py. No GPU and no transformers needed.

    python test_expert_stats.py
"""

import os
import sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn
from expert_stats import (ExpertLoadTracker, compute_lr_scales, gini, spearman, find_layers)

fails = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  <- {extra}"))
    if not cond: fails.append(name)

# ---- gini ----
check("gini uniform ~ 0", abs(gini(torch.full((64,), 10.0))) < 1e-6)
one_hot = torch.zeros(64); one_hot[0] = 640.0
check("gini collapsed ~ (n-1)/n", abs(gini(one_hot) - 63/64) < 1e-5, gini(one_hot))
check("gini all-zero = 0", gini(torch.zeros(8)) == 0.0)

# ---- spearman ----
x = torch.arange(20.0)
check("spearman monotone = 1", abs(spearman(x, 2*x+1) - 1.0) < 1e-6)
check("spearman reversed = -1", abs(spearman(x, -x) + 1.0) < 1e-6)
check("spearman constant = nan", math.isnan(spearman(x, torch.ones(20))))
check("spearman too short = nan", math.isnan(spearman(torch.tensor([1.0]), torch.tensor([2.0]))))

# ---- compute_lr_scales ----
counts = torch.tensor([0., 1., 10., 100., 1000.])
s0 = compute_lr_scales(counts, alpha=0.0)
check("alpha=0 is uniform", torch.allclose(s0, torch.ones(5)))
s1 = compute_lr_scales(counts, alpha=1.0)
check("alpha=1 finite (no inf from dead)", bool(torch.isfinite(s1).all()), s1)
check("alpha=1 dead expert left at 1.0", float(s1[0]) == 1.0, s1)
check("alpha=1 monotone over alive experts", bool((s1[1:].diff() > 0).all()), s1)
check("alpha=1 mean over alive == 1 (normalized)", abs(float(s1[1:].mean()) - 1.0) < 1e-4, s1)
s1n = compute_lr_scales(counts, alpha=1.0, normalize=False)
check("normalize=False keeps raw shares", abs(float(s1n[4]) - 4.5) < 1e-3, s1n)
# heavy-tie case: most experts dead, a few identical -> must not invent a correlation
tied = torch.tensor([0.,0.,0.,0.,5.,5.,5.,5.])
check("spearman on tied halves is finite or nan, never spurious 1.0",
      not (spearman(tied, tied.flip(0)) == 1.0), spearman(tied, tied.flip(0)))
sneg = compute_lr_scales(counts, alpha=-0.5)
check("alpha<0 finite", bool(torch.isfinite(sneg).all()), sneg)
check("alpha<0 decreasing in count", bool((sneg[1:].diff() < 0).all()), sneg)
sf = compute_lr_scales(counts, alpha=0.0, freeze_dead=True)
check("freeze_dead zeroes only dead", float(sf[0]) == 0.0 and bool((sf[1:] == 1.0).all()), sf)
big = compute_lr_scales(torch.tensor([1e-9, 1e9]), alpha=1.0)
check("clamped to [1e-3, 10]", bool(((big >= 1e-3) & (big <= 10.0)).all()), big)

# ---- ExpertLoadTracker with a fake HF-shaped model ----
E, K, T = 8, 2, 64
class FakeGate(nn.Module):
    def __init__(self):
        super().__init__(); self.num_experts = E; self.weight = nn.Parameter(torch.randn(E, 16))
    def forward(self, x):
        logits = x @ self.weight.t()
        scores, idx = torch.topk(logits, K, dim=-1)
        return logits, scores, idx            # HF 5.x router signature
class FakeMLP(nn.Module):
    def __init__(self):
        super().__init__(); self.gate = FakeGate()
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(torch.randn(E, 16, 16))
        self.experts.down_proj = nn.Parameter(torch.randn(E, 16, 16))
class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__(); self.mlp = FakeMLP()
    def forward(self, x): return self.mlp.gate(x)[0]
class FakeInner(nn.Module):
    def __init__(self, n): super().__init__(); self.layers = nn.ModuleList([FakeLayer() for _ in range(n)])
class FakeModel(nn.Module):
    def __init__(self, n=3):
        super().__init__(); self.model = FakeInner(n)
    def forward(self, x):
        for l in self.model.layers: l(x)
        return x

m = FakeModel(3)
check("find_layers finds 3", len(find_layers(m)) == 3)
tr = ExpertLoadTracker(m, top_k=K, ema=0.9)

x = torch.randn(T, 16)
tr.reset()
for _ in range(4):                      # 4 micro-steps
    tr.arm(); m(x); tr.disarm()
    m(x)                                # simulates checkpoint recompute -> must NOT count
counts = tr.counts()
check("counts for every layer", len(counts) == 3)
tot = int(counts[0].sum())
check("counts == micro_steps*T*K (recompute ignored)", tot == 4*T*K, f"{tot} vs {4*T*K}")
check("counts are exact top-k, not top-1", counts[0].sum() % K == 0)

tr.reset()
check("reset clears", len(tr.counts()) == 0)
tr.arm(); m(x); tr.disarm()
one = int(tr.counts()[0].sum())
check("one forward = T*K", one == T*K, one)

# disarmed forward adds nothing
before = tr.counts()[0].clone(); m(x)
check("disarmed forward is a no-op", torch.equal(before, tr.counts()[0]))

# EMA + shares + metrics
ema = tr.update_ema()
check("ema populated", len(ema) == 3 and ema[0].dtype == torch.float32)
sh = ExpertLoadTracker.shares(tr.counts()[0])
check("shares mean 1", abs(float(sh.mean()) - 1.0) < 1e-5)
met = tr.layer_metrics()
check("layer metrics present", {"load/gini","load/dead_expert_fraction","load/normalized_entropy"} <= set(met), met)
check("mean logits shaped [E]", tr.mean_logits()[0].shape == (E,))
tr.remove()
check("hooks removed", len(tr._handles) == 0)

# ---- fallback path: router returning bare logits ----
class BareGate(FakeGate):
    def forward(self, x): return x @ self.weight.t()
m2 = FakeModel(1); m2.model.layers[0].mlp.gate = BareGate()
m2.model.layers[0].forward = lambda x: m2.model.layers[0].mlp.gate(x)
tr2 = ExpertLoadTracker(m2, top_k=K)
tr2.reset(); tr2.arm(); m2.model.layers[0].mlp.gate(x); tr2.disarm()
check("bare-logits fallback counts T*K", int(tr2.counts()[0].sum()) == T*K, tr2.counts()[0].sum())
tr2.remove()

# ---- lr=0 really freezes (the freeze-dead mechanism) ----
for name, Opt in [("AdamW", torch.optim.AdamW)] + ([("Muon", torch.optim.Muon)] if hasattr(torch.optim, "Muon") else []):
    p_frozen = nn.Parameter(torch.randn(8, 8)); p_live = nn.Parameter(torch.randn(8, 8))
    opt = Opt([{"params": [p_frozen], "lr": 0.0}, {"params": [p_live], "lr": 0.1}],
              lr=0.1, weight_decay=0.1)
    before_f = p_frozen.detach().clone(); before_l = p_live.detach().clone()
    p_frozen.grad = torch.randn(8, 8); p_live.grad = torch.randn(8, 8)
    opt.step()
    check(f"{name}: lr=0 freezes param (incl. weight decay)", torch.equal(before_f, p_frozen))
    check(f"{name}: lr>0 moves param", not torch.equal(before_l, p_live))
print("torch.optim.Muon available:", hasattr(torch.optim, "Muon"))

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}"))
sys.exit(1 if fails else 0)

# ---- EMA vs raw deadness (freeze must key on this step, not the EMA) ----
ema_counts = torch.tensor([3.0, 50.0, 100.0])   # expert 0 has EMA history
raw_now    = torch.tensor([0.0, 50.0, 100.0])   # ...but got nothing this step
sf2 = compute_lr_scales(ema_counts, alpha=0.5, freeze_dead=True, dead_counts=raw_now)
check2 = float(sf2[0]) == 0.0 and float(sf2[1]) > 0.0
print(("PASS " if check2 else "FAIL ") + "freeze keys on raw per-step counts, not EMA" + ("" if check2 else f" <- {sf2}"))
sf3 = compute_lr_scales(ema_counts, alpha=0.5, freeze_dead=True)
noargs = float(sf3[0]) > 0.0
print(("PASS " if noargs else "FAIL ") + "without dead_counts, EMA mass keeps expert alive" + ("" if noargs else f" <- {sf3}"))
if not (check2 and noargs): sys.exit(1)
print("EMA/raw split OK")
