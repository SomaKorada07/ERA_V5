"""Train the dense ('linear') model and the MoE model, save metrics + figures.

Session 14: start from a Transformer LM whose feed-forward is one dense SwiGLU network,
then convert that block into a Mixture-of-Experts (router + N SwiGLU experts, top-k).
Both are trained on the same byte-level TinyStories data and the same token budget; we
show both keep training and reduce loss, and that the MoE routing stays balanced.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

TOKEN_BUDGET = 20_000_000
BATCH = 32

import sys
sys.path.insert(0, HERE)

from moe.data import prepare
from moe.engine import get_device, train_run
from moe.model import GPTConfig


def run():
    os.makedirs(RESULTS, exist_ok=True)
    device = get_device()
    train_bin, val_bin = prepare()
    print(f"device={device} budget={TOKEN_BUDGET:,} batch={BATCH}")

    configs = [
        ("dense", GPTConfig(ffn="dense"),
         "Dense baseline: one SwiGLU feed-forward per layer."),
        ("moe", GPTConfig(ffn="moe", routing="sparse"),
         "MoE: dense FFN replaced by 8 SwiGLU experts, top-2 routing, aux load balancing."),
    ]
    for name, cfg, note in configs:
        print(f"\n########## {name} ##########", flush=True)
        res, _ = train_run(cfg, name=name, batch_size=BATCH, token_budget=TOKEN_BUDGET,
                           train_bin=train_bin, val_bin=val_bin, device=device,
                           lr=3e-4, note=note)
        with open(os.path.join(RESULTS, f"{name}.json"), "w") as f:
            json.dump(asdict(res), f, indent=2)
        print("\n" + res.summary())
        print(f"[saved] results/{name}.json", flush=True)

    make_figures()


def _load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dense, moe = _load("dense.json"), _load("moe.json")
    if not (dense and moe):
        return

    # Fig 1: training loss vs tokens
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r, c, lab in [(dense, "#4C6EF5", f"dense ({dense['total_params']/1e6:.0f}M)"),
                      (moe, "#E8590C", f"MoE ({moe['total_params']/1e6:.0f}M total, "
                                       f"{moe['active_params']/1e6:.0f}M active)")]:
        h = r["history"]
        ax.plot([x["tokens"] / 1e6 for x in h], [x["train_loss"] for x in h], color=c, label=lab)
    ax.set_xlabel("tokens seen (millions)"); ax.set_ylabel("training loss")
    ax.set_title("Dense vs MoE — both keep training and reduce loss")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig_loss.png"), dpi=130); plt.close(fig)

    # Fig 2: routing balance over training (MoE)
    h = [x for x in moe["history"] if "max_load" in x]
    if h:
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        xs = [x["tokens"] / 1e6 for x in h]
        ax1.plot(xs, [x["max_load"] for x in h], color="#E8590C", label="busiest-expert load")
        ax1.axhline(1.0, ls="--", color="grey", lw=1, label="perfectly balanced (1.0)")
        ax1.set_xlabel("tokens seen (millions)"); ax1.set_ylabel("busiest / fair load")
        ax1.set_title("MoE load balancing: the aux loss keeps experts balanced")
        ax1.legend(loc="upper right"); ax1.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig_balance.png"), dpi=130); plt.close(fig)

    summary = {"dense": dense, "moe": moe}
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[figures + summary written]")


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        make_figures()
    else:
        run()
