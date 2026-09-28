"""Run the three assignment trainings sequentially (isolated processes) + figures.

  1. baseline      : standard residual (euler), fixed batch that runs comfortably
  2. reversible    : leapfrog reversible, SAME batch  -> memory + speed vs baseline
  3. reversible-max: leapfrog reversible, pushed to the maximum batch that fits

Each training runs in its own `python -m revllm.run_single` process so its peak-memory
number is clean, and so a failure in one does not lose the others. Then we render the
loss-curve, memory-vs-batch and speed figures used in the README.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

TOKEN_BUDGET = 50_000_000

RUNS = [
    dict(name="01_baseline",        mode="standard",   integrator="euler",
         bs=32,  lr=6e-4,  note="Baseline: standard residual (Euler h=1). Fixed batch."),
    dict(name="02_reversible",      mode="reversible", integrator="leapfrog",
         bs=32,  lr=6e-4,  note="Reversible leapfrog, SAME batch as baseline."),
    dict(name="03_reversible_max",  mode="reversible", integrator="leapfrog",
         bs=512, lr=1.5e-3, note="Reversible leapfrog at practical max batch (512) for a "
                                 "sustained run; bs=896/960 is the isolated ceiling but swaps "
                                 "with the OS + apps resident."),
]


def run_trainings():
    for r in RUNS:
        cmd = [
            sys.executable, "-m", "revllm.run_single",
            "--name", r["name"], "--mode", r["mode"], "--integrator", r["integrator"],
            "--bs", str(r["bs"]), "--token_budget", str(TOKEN_BUDGET),
            "--lr", str(r["lr"]), "--note", r["note"],
        ]
        print(f"\n########## {r['name']} ##########\n{' '.join(cmd)}", flush=True)
        env = dict(os.environ, PYTHONPATH=HERE + os.pathsep + os.environ.get("PYTHONPATH", ""))
        rc = subprocess.call(cmd, cwd=HERE, env=env)
        print(f"[{r['name']}] exit code {rc}", flush=True)


def _load(name):
    p = os.path.join(RESULTS, name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = _load("01_baseline.json")
    rev = _load("02_reversible.json")
    revmax = _load("03_reversible_max.json")
    sweep = _load("memory_sweep.json")

    # ---- Fig 1: loss curves (baseline vs reversible, same batch) ----
    if base and rev:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for res, c, lab in [(base, "#4C6EF5", "baseline (standard/euler)"),
                            (rev, "#E8590C", "reversible (leapfrog)")]:
            xs = [h["tokens"] / 1e6 for h in res["loss_history"]]
            ys = [h["train_loss"] for h in res["loss_history"]]
            ax.plot(xs, ys, label=lab, color=c, lw=1.8)
        ax.set_xlabel("tokens seen (millions)")
        ax.set_ylabel("training loss")
        ax.set_title("20M LLM on TinyStories — same batch (32), 50M tokens")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig_loss.png"), dpi=130)
        plt.close(fig)

    # ---- Fig 2: memory vs batch (standard vs reversible) ----
    if sweep:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for key, c, lab in [("standard", "#4C6EF5", "standard"),
                            ("reversible", "#E8590C", "reversible")]:
            pts = [(r["batch_size"], r["peak_gb"]) for r in sweep[key] if r["peak_gb"]]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=c, label=lab)
        ax.axhline(68.7, ls="--", color="grey", lw=1, label="68.7 GB physical")
        ax.set_xlabel("batch size (seq len 512)")
        ax.set_ylabel("peak memory (GB, driver high-water)")
        ax.set_title("Reversibility: activation memory ~ independent of depth")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig_memory.png"), dpi=130)
        plt.close(fig)

    # ---- summary.json ----
    summary = {k: _load(f"{k}.json") for k in ["01_baseline", "02_reversible", "03_reversible_max"]}
    summary["memory_sweep"] = sweep
    summary["reconstruction"] = _load("reconstruction.json")
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[figures + summary written]")


if __name__ == "__main__":
    if "--figures-only" not in sys.argv:
        run_trainings()
    make_figures()
