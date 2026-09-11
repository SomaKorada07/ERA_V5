"""
part1_2_adam_by_hand.py

Part 1: Reproduce Adam by hand on a single weight fed five gradients. Compute
        m, v, m_hat, v_hat and the step manually, then check every quantity
        against torch.optim.Adam.

Part 2: Turn bias correction OFF and compare the first twenty steps to the
        bias-corrected version. Plot both trajectories and report the step after
        which the difference stops mattering.
"""

import json
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Adam hyperparameters
LR = 1e-2
BETA1, BETA2 = 0.9, 0.999
EPS = 1e-8

# One weight, five gradients (chosen to have mixed signs and magnitudes).
W0 = 0.500000
GRADS = [0.100, -0.200, 0.050, 0.400, -0.150]


def adam_by_hand(w0, grads, lr, b1, b2, eps, bias_correction=True):
    """Pure-Python Adam. Returns per-step dict of every intermediate quantity."""
    w = w0
    m = 0.0
    v = 0.0
    rows = []
    for t, g in enumerate(grads, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        if bias_correction:
            m_hat = m / (1 - b1 ** t)
            v_hat = v / (1 - b2 ** t)
        else:
            m_hat, v_hat = m, v
        step = lr * m_hat / (math.sqrt(v_hat) + eps)
        w = w - step
        rows.append(dict(t=t, g=g, m=m, v=v, m_hat=m_hat, v_hat=v_hat,
                         step=step, w=w))
    return rows


def adam_pytorch(w0, grads, lr, b1, b2, eps):
    """torch.optim.Adam driven with the same scalar gradients, one step each."""
    w = torch.tensor([w0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr, betas=(b1, b2), eps=eps)
    rows = []
    for t, g in enumerate(grads, start=1):
        opt.zero_grad()
        w.grad = torch.tensor([g], dtype=torch.float64)
        opt.step()
        state = opt.state[w]
        rows.append(dict(
            t=t,
            m=state["exp_avg"].item(),
            v=state["exp_avg_sq"].item(),
            w=w.item(),
        ))
    return rows


def part1():
    print("=" * 78)
    print("PART 1 — Adam by hand vs PyTorch")
    print("=" * 78)
    print(f"w0 = {W0}   lr = {LR}   betas = ({BETA1}, {BETA2})   eps = {EPS}")
    print(f"gradients = {GRADS}\n")

    hand = adam_by_hand(W0, GRADS, LR, BETA1, BETA2, EPS, bias_correction=True)
    torch_rows = adam_pytorch(W0, GRADS, LR, BETA1, BETA2, EPS)

    hdr = (f"{'t':>2} {'g':>7} {'m':>11} {'v':>12} {'m_hat':>11} "
           f"{'v_hat':>12} {'step':>12} {'w_hand':>11} {'w_torch':>11} {'|Δw|':>10}")
    print(hdr)
    print("-" * len(hdr))
    max_dev = 0.0
    report = []
    for h, tr in zip(hand, torch_rows):
        dw = abs(h["w"] - tr["w"])
        dm = abs(h["m"] - tr["m"])
        dv = abs(h["v"] - tr["v"])
        max_dev = max(max_dev, dw, dm, dv)
        print(f"{h['t']:>2} {h['g']:>7.3f} {h['m']:>11.7f} {h['v']:>12.8f} "
              f"{h['m_hat']:>11.7f} {h['v_hat']:>12.8f} {h['step']:>12.8f} "
              f"{h['w']:>11.7f} {tr['w']:>11.7f} {dw:>10.2e}")
        report.append(dict(t=h["t"], **{k: h[k] for k in
                     ("g", "m", "v", "m_hat", "v_hat", "step", "w")},
                     w_torch=tr["w"], abs_dw=dw, abs_dm=dm, abs_dv=dv))

    print("-" * len(hdr))
    print(f"max |hand - torch| across m, v, w over all steps = {max_dev:.3e}")
    ok = max_dev < 1e-10
    print(f"agreement to <1e-10: {'PASS' if ok else 'FAIL'}\n")
    assert ok, "Hand-computed Adam disagreed with PyTorch."
    return report


def part2():
    print("=" * 78)
    print("PART 2 — Bias correction ON vs OFF, first 20 steps")
    print("=" * 78)

    # A longer, fixed gradient stream so the two versions can be compared over
    # 20 steps. Same gradient each step keeps the effect of bias correction
    # (the 1/(1-beta^t) factors) clean and easy to read.
    n = 20
    rng = np.random.default_rng(0)
    grads = list(0.1 + 0.02 * rng.standard_normal(n))  # ~constant, small noise

    on = adam_by_hand(W0, grads, LR, BETA1, BETA2, EPS, bias_correction=True)
    off = adam_by_hand(W0, grads, LR, BETA1, BETA2, EPS, bias_correction=False)

    steps = np.arange(1, n + 1)
    w_on = np.array([r["w"] for r in on])
    w_off = np.array([r["w"] for r in off])
    step_on = np.array([r["step"] for r in on])
    step_off = np.array([r["step"] for r in off])
    rel = np.abs(step_on - step_off) / np.abs(step_on)

    # The per-step ratio uncorrected/corrected is EXACT and gradient-independent
    # (eps aside):   step_off/step_on = (1 - beta1^t) / sqrt(1 - beta2^t).
    # "Difference stops mattering" = that ratio is within `tol` of 1.
    def bias_ratio(t):
        return (1 - BETA1 ** t) / math.sqrt(1 - BETA2 ** t)

    def settle_at(tol):
        for t in range(1, 100000):
            if abs(bias_ratio(t) - 1.0) < tol:
                return t
        return None

    settle = {tol: settle_at(tol) for tol in (0.10, 0.05, 0.01)}

    print(f"{'t':>3} {'step_on':>12} {'step_off':>12} {'off/on':>9} "
          f"{'(1-b1^t)/√(1-b2^t)':>19} {'w_on':>11} {'w_off':>11}")
    print("-" * 82)
    for i in range(n):
        print(f"{steps[i]:>3} {step_on[i]:>12.8f} {step_off[i]:>12.8f} "
              f"{step_off[i]/step_on[i]:>9.3f} {bias_ratio(steps[i]):>19.3f} "
              f"{w_on[i]:>11.7f} {w_off[i]:>11.7f}")
    print("-" * 82)
    print("Empirical off/on matches the closed form (1-b1^t)/sqrt(1-b2^t) exactly.")
    print(f"Within the plotted 20 steps the update is still 3–7x too large — it "
          f"does NOT settle here.")
    print("The difference stops mattering on the beta2 timescale:")
    for tol, t in settle.items():
        print(f"    within {tol:>4.0%}:  t = {t}")
    print(f"beta1 alone would settle by t={math.ceil(math.log(0.01)/math.log(BETA1))} "
          f"(1/(1-b1^t) within 1%), but beta2={BETA2} dominates the denominator.\n")
    settle_step = settle[0.01]
    t_m = math.ceil(math.log(0.01) / math.log(BETA1))
    t_v = math.ceil(math.log(0.01) / math.log(BETA2))

    # Plot: trajectories + updates over 20 steps, and the ratio out to settle.
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    ax1.plot(steps, w_on, "o-", label="bias correction ON", color="#2563eb")
    ax1.plot(steps, w_off, "s--", label="bias correction OFF", color="#dc2626")
    ax1.set_xlabel("step t"); ax1.set_ylabel("weight w"); ax1.set_title("Weight trajectory (20 steps)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(steps, step_on, "o-", label="update ON", color="#2563eb")
    ax2.plot(steps, step_off, "s--", label="update OFF", color="#dc2626")
    ax2.set_xlabel("step t"); ax2.set_ylabel("update size |Δw|")
    ax2.set_title("Per-step update (still 3–7x off)"); ax2.legend(); ax2.grid(alpha=0.3)

    tt = np.arange(1, settle[0.01] + 200)
    ratio = np.array([bias_ratio(int(t)) for t in tt])
    ax3.plot(tt, ratio, color="#7c3aed")
    ax3.axhline(1.0, color="k", lw=0.8)
    for tol, col in [(0.10, "#f59e0b"), (0.05, "#16a34a"), (0.01, "#0891b2")]:
        ax3.axvline(settle[tol], color=col, ls=":", lw=2,
                    label=f"within {tol:.0%}: t={settle[tol]}")
    ax3.set_xscale("log")
    ax3.set_xlabel("step t (log)"); ax3.set_ylabel("update OFF / update ON")
    ax3.set_title("When the difference stops mattering"); ax3.legend(); ax3.grid(alpha=0.3)

    fig.suptitle("Part 2 — bias correction ON vs OFF (Adam)")
    fig.tight_layout()
    fig.savefig("assets/part2_bias_correction.png", dpi=130)
    print("saved assets/part2_bias_correction.png")
    return dict(settle_1pct=settle[0.01], settle_5pct=settle[0.05],
                settle_10pct=settle[0.10], t_m=t_m, t_v=t_v)


if __name__ == "__main__":
    p1 = part1()
    p2 = part2()
    with open("assets/part1_2_results.json", "w") as f:
        json.dump({"part1": p1, "part2": p2}, f, indent=2)
    print("\nsaved assets/part1_2_results.json")
