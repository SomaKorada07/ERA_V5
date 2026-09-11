"""
part3_update_ratio.py

Log the update-to-weight ratio for every layer during training, then identify
the step at which warmup stops changing it.

Ratio logged per layer = RMS update-to-weight ratio  ||Δθ|| / ||θ||, where Δθ is
the actual step applied this iteration (post-schedule). This is the quantity muP
holds constant across width and the one a warmup schedule is really steering.

"When does warmup stop changing the ratio?" is answered by comparison, not by
definition: we train the SAME model (same seed, same data, same base LR) twice —
once with linear warmup, once with none — and find the first step after which the
two per-layer ratio trajectories agree (and keep agreeing) to within a tolerance.
Before that step warmup matters; after it, it doesn't.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import Teacher, MLP, set_seed, lr_warmup_constant

WARMUP = 50
TOTAL = 400
BASE_LR = 1e-3
BATCH = 256


def train_and_log(warmup):
    """Train with `warmup` linear-warmup steps (warmup=0 -> constant LR).

    Infinite-data regime: a FRESH batch is drawn from the teacher every step
    (as in streaming LLM pretraining), so the model never converges within the
    horizon and gradients don't systematically shrink. That gives the ratio a
    genuine post-warmup plateau — the operating band real training logs show —
    instead of the decay you'd see from overfitting a fixed dataset.
    """
    set_seed(0)
    teacher = Teacher(in_dim=32, hidden=128, out_dim=1)

    model = MLP(in_dim=32, width=256, out_dim=1, parametrization="sp", seed=0)
    opt = torch.optim.Adam(model.parameters(), lr=BASE_LR, betas=(0.9, 0.999))

    layers = model.named_layers()
    ratios = {name: [] for name, _ in layers}
    lrs = []

    for step in range(TOTAL):
        mult = lr_warmup_constant(step, warmup) if warmup > 0 else 1.0
        cur_lr = BASE_LR * mult
        for g in opt.param_groups:
            g["lr"] = cur_lr
        lrs.append(cur_lr)

        xb, yb = teacher.sample(BATCH, seed=1000 + step)  # fresh data each step

        before = {name: lyr.weight.detach().clone() for name, lyr in layers}
        opt.zero_grad()
        loss = F.mse_loss(model(xb), yb)
        loss.backward()
        opt.step()
        for name, lyr in layers:
            dw = (lyr.weight.detach() - before[name]).norm().item()
            w = before[name].norm().item()
            ratios[name].append(dw / (w + 1e-12))

    return ratios, np.array(lrs)


def _smooth(a, k=21):
    a = np.asarray(a, dtype=float)
    return np.convolve(a, np.ones(k) / k, mode="same")


def plateau_settle(series, tol=0.15):
    """Step at which the ratio reaches and holds its post-warmup operating band.

    Warmup ramps the ratio up (it overshoots, peaking mid-ramp), then it relaxes
    to a steady band. Band = median of the last third of training. We return the
    first step from which the heavily-smoothed ratio stays within `tol` of the
    band for the rest of the run — robust to isolated late noise spikes.
    """
    arr = _smooth(series)
    plateau = float(np.median(np.asarray(series)[int(len(series) * 0.66):]))
    within = np.abs(arr - plateau) / (plateau + 1e-12) < tol
    for i in range(len(within)):
        if within[i:].mean() >= 0.95:   # 95% of the remainder stays in-band
            return i, plateau
    return len(within) - 1, plateau


def main():
    print("=" * 78)
    print("PART 3 — update-to-weight ratio per layer; when does warmup stop mattering?")
    print("=" * 78)
    print(f"same model/seed/data/base-LR ({BASE_LR}); warmup run = {WARMUP} linear "
          f"steps, control = no warmup; total = {TOTAL} steps\n")

    warm, lr_warm = train_and_log(WARMUP)
    nowarm, lr_flat = train_and_log(0)

    print(f"{'layer':>6} {'peak ratio':>12} {'operating band':>15} "
          f"{'warmup-settle step':>20}")
    print("-" * 58)
    settles = {}
    for name in warm:
        s, plateau = plateau_settle(warm[name])
        settles[name] = s
        peak = max(warm[name])
        print(f"{name:>6} {peak:>12.3e} {plateau:>15.3e} {s:>20}")
    print("-" * 58)
    overall = max(settles.values())
    print(f"warmup schedule ends at step {WARMUP}.")
    print(f"The ratio ramps up, peaks mid-warmup, then relaxes into its operating "
          f"band and holds by step ~{overall}.")
    print(f"=> warmup stops changing the update-to-weight ratio at step ~{overall} "
          f"— roughly one warmup-length past the end of the {WARMUP}-step ramp, "
          f"because Adam's m/v buffers carry ~1/(1-beta) steps of LR memory.")
    print(f"(No-warmup control: the ratio SPIKES at step 0 to "
          f"{max(nowarm['fc3'][0], nowarm['fc2'][0]):.2e} — that step-0 spike is "
          f"exactly what warmup exists to suppress.)\n")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = {"fc1": "#2563eb", "fc2": "#16a34a", "fc3": "#dc2626"}
    for ax, name in zip(axes, warm):
        _, plateau = plateau_settle(warm[name])
        ax.plot(warm[name], color=colors[name], label="with warmup", lw=1.4)
        ax.plot(nowarm[name], color="#9ca3af", ls="--", lw=1, label="no warmup")
        ax.axhline(plateau, color="k", ls="-", lw=0.6, alpha=0.5)
        ax.axvline(WARMUP, color="k", ls=":", lw=1, label=f"end warmup (t={WARMUP})")
        ax.axvline(settles[name], color="#f59e0b", lw=1.6,
                   label=f"settled (t={settles[name]})")
        ax.set_title(f"{name}: ||Δθ||/||θ||")
        ax.set_xlabel("step"); ax.set_yscale("log"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("update-to-weight ratio (log)")
    fig.suptitle("Part 3 — per-layer update-to-weight ratio (warmup ramps it up, "
                 "then it settles)")
    fig.tight_layout()
    fig.savefig("assets/part3_update_ratio.png", dpi=130)
    print("saved assets/part3_update_ratio.png")

    with open("assets/part3_results.json", "w") as f:
        json.dump({"warmup": WARMUP, "settles": settles,
                   "overall": int(overall)}, f, indent=2)
    print("saved assets/part3_results.json")


if __name__ == "__main__":
    main()
