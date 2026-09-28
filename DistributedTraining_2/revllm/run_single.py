"""Run ONE training experiment in an isolated process and dump results/<name>.json.

Isolation matters for the peak-memory number: MPS driver memory is a high-water
mark that does not reset within a process, so each experiment gets a fresh Python
process. Invoked by run_experiments.py.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from .data import prepare
from .engine import get_device, train_run
from .model import GPTConfig

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="standard")
    ap.add_argument("--integrator", default="euler")
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--token_budget", type=int, default=50_000_000)
    ap.add_argument("--n_layer", type=int, default=9)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--step_size", type=float, default=0.5)
    ap.add_argument("--loss_chunk", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    train_bin, val_bin = prepare()

    cfg = GPTConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        block_size=args.block_size, mode=args.mode, integrator=args.integrator,
        step_size=args.step_size, loss_chunk=args.loss_chunk,
    )
    print(f"=== {args.name} | mode={args.mode}/{args.integrator} bs={args.bs} "
          f"budget={args.token_budget:,} device={device} ===", flush=True)

    res = train_run(
        cfg, name=args.name, batch_size=args.bs, token_budget=args.token_budget,
        train_bin=train_bin, val_bin=val_bin, device=device, lr=args.lr,
        log_every=args.log_every,
    )
    res.note = args.note
    out = os.path.join(RESULTS_DIR, f"{args.name}.json")
    with open(out, "w") as f:
        json.dump(asdict(res), f, indent=2)
    print("\n" + res.summary())
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
