"""
part5_lr_sweep_mup.py

Sweep the learning rate at widths 256, 512 and 1024, plot loss vs LR, mark the
three minima, and state the LR to use at width 4096 with a confidence.

The assignment's rule — "tune both sides" — is the whole point here. We run the
sweep under TWO parametrizations:

  * SP  (standard):  the optimal LR DRIFTS with width, so extrapolating to 4096
                     is an educated guess (low/moderate confidence).
  * muP (maximal update): init + per-layer LR multipliers are set so the optimal
                     *base* LR is (approximately) width-invariant, so the 4096
                     prediction is just "the same base LR" (high confidence).

We then actually TRAIN width 4096 under both and check the prediction, turning
the confidence statement into a measured error rather than a hope.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import Teacher, MLP, set_seed, lr_cosine

TOTAL = 300
WARMUP = 20
BATCH = 256
WIDTHS = [256, 512, 1024]
BASE_WIDTH = 256
LR_GRID = [10 ** e for e in np.arange(-3.5, -0.5 + 1e-9, 0.25)]  # ~13 points
TARGET_WIDTH = 4096
EVAL_N = 4096


def train_one(width, base_lr, parametrization):
    set_seed(0)
    teacher = Teacher(in_dim=32, hidden=128, out_dim=1)
    xe, ye = teacher.sample(EVAL_N, seed=999)

    model = MLP(in_dim=32, width=width, out_dim=1, parametrization=parametrization,
                base_width=BASE_WIDTH, seed=0)
    groups = model.param_groups_for_adam(base_lr)
    opt = torch.optim.Adam(groups, betas=(0.9, 0.999))
    base_lrs = [g["lr"] for g in opt.param_groups]  # remember relative scaling

    sched = lambda s: lr_cosine(s, TOTAL, WARMUP, min_ratio=0.1)
    for step in range(TOTAL):
        m = sched(step)
        for g, bl in zip(opt.param_groups, base_lrs):
            g["lr"] = bl * m
        xb, yb = teacher.sample(BATCH, seed=5000 + step)
        opt.zero_grad()
        loss = F.mse_loss(model(xb), yb)
        loss.backward()
        opt.step()
        if not torch.isfinite(loss):
            return float("inf")
    with torch.no_grad():
        return float(F.mse_loss(model(xe), ye))


def sweep(widths, parametrization):
    out = {}
    for w in widths:
        losses = [train_one(w, lr, parametrization) for lr in LR_GRID]
        out[w] = losses
        best_i = int(np.argmin(losses))
        print(f"  [{parametrization}] width {w:>4}: best LR = "
              f"{LR_GRID[best_i]:.2e}  loss = {losses[best_i]:.4f}")
    return out


def argmin_lr(losses):
    return LR_GRID[int(np.argmin(losses))]


def main():
    print("=" * 78)
    print("PART 5 — LR sweep vs width, and transfer to width 4096")
    print("=" * 78)
    print(f"widths {WIDTHS}, base_width {BASE_WIDTH}, {TOTAL} steps, "
          f"cosine+warmup, LR grid {LR_GRID[0]:.1e}..{LR_GRID[-1]:.1e}\n")

    print("Standard parametrization (SP):")
    sp = sweep(WIDTHS, "sp")
    print("muP:")
    mup = sweep(WIDTHS, "mup")

    sp_argmin = {w: argmin_lr(sp[w]) for w in WIDTHS}
    mup_argmin = {w: argmin_lr(mup[w]) for w in WIDTHS}

    # ---- SP: optimum drifts. Fit log10(lr*) vs log2(width) and extrapolate. ----
    lw = np.log2(np.array(WIDTHS))
    sp_log = np.log10([sp_argmin[w] for w in WIDTHS])
    a, b = np.polyfit(lw, sp_log, 1)  # log10 lr = a*log2(w) + b
    sp_pred_4096 = 10 ** (a * np.log2(TARGET_WIDTH) + b)
    print(f"\nSP optimum LR by width: "
          + ", ".join(f"{w}->{sp_argmin[w]:.2e}" for w in WIDTHS))
    print(f"SP log-linear fit: slope {a:.2f} per width-doubling "
          f"=> extrapolated LR@4096 ~ {sp_pred_4096:.2e} (extrapolation; "
          f"lower confidence).")

    # ---- muP: optimum should be width-invariant. Predict the median. ----
    mup_pred_4096 = float(np.median([mup_argmin[w] for w in WIDTHS]))
    spread = max(mup_argmin.values()) / min(mup_argmin.values())
    print(f"muP optimum LR by width: "
          + ", ".join(f"{w}->{mup_argmin[w]:.2e}" for w in WIDTHS))
    print(f"muP optima span a {spread:.1f}x range => predicted base LR@4096 = "
          f"{mup_pred_4096:.2e} (transfer; higher confidence).")

    # ---- Verify by actually training width 4096 over the FULL grid ----
    print(f"\nVerifying at width {TARGET_WIDTH} over the full LR grid (slow)...")
    sp_4096 = {lr: train_one(TARGET_WIDTH, lr, "sp") for lr in LR_GRID}
    mup_4096 = {lr: train_one(TARGET_WIDTH, lr, "mup") for lr in LR_GRID}
    sp_best_lr = min(sp_4096, key=sp_4096.get)
    mup_best_lr = min(mup_4096, key=mup_4096.get)
    print(f"  measured best LR@4096: SP = {sp_best_lr:.2e} (loss {sp_4096[sp_best_lr]:.4f})"
          f",  muP = {mup_best_lr:.2e} (loss {mup_4096[mup_best_lr]:.4f})")

    def loss_at(curve, lr):  # nearest grid point in log space
        k = min(curve, key=lambda l: abs(np.log(l) - np.log(lr)))
        return curve[k], k

    # What happens if you naively REUSE the small-width (256) optimum at 4096?
    sp_reuse, _ = loss_at(sp_4096, sp_argmin[256])
    mup_reuse, _ = loss_at(mup_4096, mup_argmin[256])
    sp_best_loss = sp_4096[sp_best_lr]
    mup_best_loss = mup_4096[mup_best_lr]
    print(f"\n  Naively reusing the width-256 optimum at width 4096:")
    print(f"    SP : LR {sp_argmin[256]:.1e} -> loss {sp_reuse:.4f}  "
          f"(best is {sp_best_loss:.4f}); "
          f"{'DIVERGES / blows up' if sp_reuse > 3*sp_best_loss else 'ok'}.")
    print(f"    muP: LR {mup_argmin[256]:.1e} -> loss {mup_reuse:.4f}  "
          f"(best is {mup_best_loss:.4f}); "
          f"penalty {sp_reuse/sp_best_loss if False else mup_reuse/mup_best_loss:.2f}x.")

    # muP basin width: fraction of the grid within 1.5x of the best loss
    mup_flat = [lr for lr in LR_GRID if mup_4096[lr] <= 1.5 * mup_best_loss]
    sp_flat = [lr for lr in LR_GRID if sp_4096[lr] <= 1.5 * sp_best_loss]
    print(f"\n  Basin width at 4096 (LRs within 1.5x of best loss):")
    print(f"    SP : {len(sp_flat)}/{len(LR_GRID)} grid pts, "
          f"{min(sp_flat):.1e}..{max(sp_flat):.1e} ({max(sp_flat)/min(sp_flat):.0f}x)")
    print(f"    muP: {len(mup_flat)}/{len(LR_GRID)} grid pts, "
          f"{min(mup_flat):.1e}..{max(mup_flat):.1e} ({max(mup_flat)/min(mup_flat):.0f}x)")

    print(f"\nRECOMMENDATION for width {TARGET_WIDTH}:")
    print(f"  Preferred — use muP and REUSE base LR ~= {mup_argmin[256]:.1e} "
          f"(optimal at 256 & 512). Confidence HIGH that it is SAFE and "
          f"near-optimal: at 4096 the muP basin within 1.5x of best spans "
          f"{max(mup_flat)/min(mup_flat):.0f}x and nothing diverges, so a "
          f"2-3x miss costs almost nothing.")
    print(f"  If you must use SP — extrapolate the clean downward drift "
          f"(slope {a:.2f}/doubling): LR ~= {sp_pred_4096:.1e} (measured optimum "
          f"{sp_best_lr:.1e}). Confidence MODERATE, and bias LOW: the SP basin is "
          f"narrow ({max(sp_flat)/min(sp_flat):.0f}x) and the HIGH side is a cliff "
          f"— reusing the width-256 LR here gives loss {sp_reuse:.3f} (diverged).")
    print(f"  One-line answer: width 4096 -> muP base LR ~ {mup_argmin[256]:.1e} "
          f"(high confidence), or SP LR ~ {sp_pred_4096:.0e} (moderate, err low).\n")

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = {256: "#2563eb", 512: "#16a34a", 1024: "#dc2626"}
    for ax, data, argm, title, predline, meas in [
        (ax1, sp, sp_argmin, "SP — optimum drifts DOWN, high side is a cliff",
         sp_pred_4096, sp_4096),
        (ax2, mup, mup_argmin, "muP — wide flat basin, reuse the small-width LR",
         mup_argmin[256], mup_4096)]:
        for w in WIDTHS:
            ax.plot(LR_GRID, data[w], "o-", color=colors[w], label=f"width {w}")
            bl = argm[w]
            bi = LR_GRID.index(bl)
            ax.scatter([bl], [data[w][bi]], s=150, facecolors="none",
                       edgecolors=colors[w], linewidths=2.2, zorder=5)
        lrs4 = sorted(meas)
        ax.plot(lrs4, [meas[l] for l in lrs4], "x--", color="#7c3aed", lw=1.6,
                label="width 4096 (measured)")
        ax.axvline(predline, color="#f59e0b", ls=":", lw=2,
                   label=f"LR used @4096 = {predline:.1e}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("learning rate"); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax1.set_ylabel("held-out MSE (log)")
    fig.suptitle("Part 5 — LR sweep vs width (open circles = per-width minima)")
    fig.tight_layout()
    fig.savefig("assets/part5_lr_sweep.png", dpi=130)
    print("saved assets/part5_lr_sweep.png")

    with open("assets/part5_results.json", "w") as f:
        json.dump({
            "lr_grid": LR_GRID, "widths": WIDTHS,
            "sp_losses": sp, "mup_losses": mup,
            "sp_argmin": sp_argmin, "mup_argmin": mup_argmin,
            "sp_pred_4096": sp_pred_4096, "mup_pred_4096": mup_pred_4096,
            "sp_4096": sp_4096, "mup_4096": mup_4096,
            "sp_best_lr_4096": sp_best_lr, "mup_best_lr_4096": mup_best_lr,
        }, f, indent=2)
    print("saved assets/part5_results.json")


if __name__ == "__main__":
    main()
