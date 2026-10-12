"""Train the dense ('linear') model and the MoE model, save metrics + figures.

Session 14: start from a Transformer LM whose feed-forward is one dense SwiGLU network,
then convert that block into a Mixture-of-Experts (router + N SwiGLU experts, top-k).
Both are trained on the same byte-level TinyStories data and the same token budget; we
show both keep training and reduce loss, and that the MoE routing stays balanced.

On top of that from-scratch comparison, we run the actual conversion the assignment asks
for (Expert-Upcycling, see class notes): take the *trained* dense checkpoint, copy its
SwiGLU feed-forward into every expert (`moe.engine._upcycle`), verify the conversion
didn't blow up the loss, then continue training two matched branches from that same
checkpoint for an equal extra token budget — "dense_continued" (keep training the dense
model) vs "moe_upcycled" (the converted MoE model) — which is the comparison the sparse-
upcycling literature actually makes (does converting beat just continuing to train the
dense model for the same extra compute?).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

TOKEN_BUDGET = 20_000_000       # from-scratch dense and MoE baselines
CONTINUE_BUDGET = 10_000_000    # extra budget each continued branch gets after the handoff
BATCH = 32

import sys
sys.path.insert(0, HERE)

import torch

from moe.data import prepare
from moe.engine import evaluate, get_device, train_run, _upcycle
from moe.model import GPT, GPTConfig


def run():
    os.makedirs(RESULTS, exist_ok=True)
    device = get_device()
    train_bin, val_bin = prepare()
    print(f"device={device} budget={TOKEN_BUDGET:,} batch={BATCH}")

    configs = [
        ("dense", GPTConfig(ffn="dense"),
         "Dense baseline: one SwiGLU feed-forward per layer."),
        ("moe", GPTConfig(ffn="moe", routing="sparse"),
         "MoE from scratch (random init): dense FFN replaced by 8 SwiGLU experts, "
         "top-2 routing, aux load balancing."),
    ]
    dense_model = None
    for name, cfg, note in configs:
        print(f"\n########## {name} ##########", flush=True)
        res, model = train_run(cfg, name=name, batch_size=BATCH, token_budget=TOKEN_BUDGET,
                               train_bin=train_bin, val_bin=val_bin, device=device,
                               lr=3e-4, note=note)
        if name == "dense":
            dense_model = model  # trained checkpoint, handed off to the upcycle experiment
            torch.save(model.state_dict(), os.path.join(RESULTS, "dense_checkpoint.pt"))
        with open(os.path.join(RESULTS, f"{name}.json"), "w") as f:
            json.dump(asdict(res), f, indent=2)
        print("\n" + res.summary())
        print(f"[saved] results/{name}.json", flush=True)

    run_upcycle_experiment(dense_model, train_bin, val_bin, device)
    make_figures()


def run_upcycle_experiment(dense_model, train_bin, val_bin, device):
    """Expert-Upcycling: convert the trained dense checkpoint into an MoE model, verify
    the conversion loss, then continue two matched branches (dense_continued vs
    moe_upcycled) from that same checkpoint for an equal extra token budget.
    """
    from moe.data import ByteData

    print("\n########## upcycle: dense checkpoint -> MoE ##########", flush=True)
    dense_cfg = dense_model.cfg
    # _upcycle copies each dense SwiGLU FFN whole into every expert, so expert_width must
    # equal the dense d_ff (unlike the from-scratch "moe" config above, which narrows
    # expert_width to d_ff // top_k so its active params match the dense model). This
    # mirrors the literature: a sparse-upcycled expert IS a full copy of the FFN, so the
    # upcycled model's active compute is higher than the dense model's by design.
    moe_cfg = GPTConfig(ffn="moe", routing="sparse", expert_width=dense_cfg.d_ff)
    val = ByteData(val_bin)

    # 1) checkpoint handoff: upcycle the trained dense model into a same-seed MoE model
    #    (copy shared weights + dense FFN -> every expert; router stays random init).
    torch.manual_seed(1337)
    moe_upcycled = GPT(moe_cfg).to(device)
    _upcycle(moe_upcycled, dense_model)

    # 2) verify the conversion loss: right after the copy (no MoE-specific training yet)
    #    the upcycled model's val loss should sit close to the dense checkpoint's loss,
    #    far below a random-init MoE of the same shape -- i.e. the conversion actually
    #    transferred what the dense model learned instead of discarding it.
    torch.manual_seed(1337)
    moe_random = GPT(moe_cfg).to(device)
    dense_val = evaluate(dense_model, val, BATCH, dense_cfg.block_size, device)
    moe_random_val = evaluate(moe_random, val, BATCH, moe_cfg.block_size, device)
    moe_upcycled_val = evaluate(moe_upcycled, val, BATCH, moe_cfg.block_size, device)
    conversion = {
        "dense_checkpoint_val_loss": dense_val,
        "moe_random_init_val_loss": moe_random_val,
        "moe_upcycled_val_loss": moe_upcycled_val,
    }
    print(f"[upcycle check] dense ckpt {dense_val:.4f}  ->  upcycled MoE {moe_upcycled_val:.4f}"
          f"  (random-init MoE would start at {moe_random_val:.4f})")
    assert moe_upcycled_val < moe_random_val, (
        "upcycled MoE should start far below a random-init MoE of the same shape")
    with open(os.path.join(RESULTS, "upcycle_check.json"), "w") as f:
        json.dump(conversion, f, indent=2)
    print("[saved] results/upcycle_check.json", flush=True)

    # 3) continue matched branches from the same checkpoint for the same extra budget.
    branches = [
        ("dense_continued", dense_cfg, dense_model,
         "Dense checkpoint, continued training for the same extra budget (no conversion)."),
        ("moe_upcycled", moe_cfg, moe_upcycled,
         "Same dense checkpoint, converted to MoE (Expert-Upcycling), continued training."),
    ]
    for name, cfg, init_model, note in branches:
        print(f"\n########## {name} ##########", flush=True)
        res, _ = train_run(cfg, name=name, batch_size=BATCH, token_budget=CONTINUE_BUDGET,
                            train_bin=train_bin, val_bin=val_bin, device=device, lr=3e-4,
                            note=note, init_from=init_model)
        with open(os.path.join(RESULTS, f"{name}.json"), "w") as f:
            json.dump(asdict(res), f, indent=2)
        print("\n" + res.summary())
        print(f"[saved] results/{name}.json", flush=True)


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

    # Fig 3 + summary: upcycle experiment, if it was run
    upcycle_check = _load("upcycle_check.json")
    dense_cont, moe_up = _load("dense_continued.json"), _load("moe_upcycled.json")
    if upcycle_check:
        summary["upcycle_check"] = upcycle_check
    if dense_cont and moe_up:
        summary["dense_continued"] = dense_cont
        summary["moe_upcycled"] = moe_up
        fig, ax = plt.subplots(figsize=(7, 4.5))
        offset = dense["tokens_seen"] / 1e6  # both branches continue from this checkpoint
        if upcycle_check:
            ax.scatter([offset], [upcycle_check["dense_checkpoint_val_loss"]],
                       color="#4C6EF5", zorder=5, label="dense checkpoint (handoff point)")
            ax.scatter([offset], [upcycle_check["moe_upcycled_val_loss"]],
                       color="#E8590C", marker="x", zorder=5, label="upcycled MoE (just converted)")
        for r, c, lab in [(dense_cont, "#4C6EF5", "dense, continued (no conversion)"),
                          (moe_up, "#E8590C", "MoE, upcycled from checkpoint, continued")]:
            h = r["history"]
            ax.plot([offset + x["tokens"] / 1e6 for x in h], [x["train_loss"] for x in h],
                    color=c, label=lab)
        ax.axvline(offset, ls="--", color="grey", lw=1)
        ax.set_xlabel("tokens seen (millions)"); ax.set_ylabel("training loss")
        ax.set_title("Upcycling vs continued dense training, same checkpoint + budget")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig_upcycle.png"), dpi=130); plt.close(fig)

    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[figures + summary written]")


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        make_figures()
    else:
        run()
