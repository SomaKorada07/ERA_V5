"""Measure peak memory + throughput for ONE (mode, batch) config, as a subprocess.

Run in isolation so an MPS kernel-size abort (SIGABRT) or OOM kills only this probe,
not the parent sweep. Prints a single JSON line on success; non-zero exit on failure.
"""
from __future__ import annotations

import argparse
import json
import sys

from .engine import get_device, measure_step_memory
from .model import GPTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="standard")
    ap.add_argument("--integrator", default="euler")
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--n_layer", type=int, default=9)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--step_size", type=float, default=0.5)
    ap.add_argument("--loss_chunk", type=int, default=16384)
    args = ap.parse_args()

    dev = get_device()
    cfg = GPTConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        block_size=args.block_size, mode=args.mode, integrator=args.integrator,
        step_size=args.step_size, loss_chunk=args.loss_chunk,
    )
    peak, tps = measure_step_memory(cfg, args.bs, dev, warmup=2, iters=4)
    print(json.dumps({
        "ok": True, "mode": args.mode, "bs": args.bs,
        "peak_mem_bytes": int(peak), "tokens_per_sec": float(tps),
        "tokens_per_step": args.bs * args.block_size, "device": dev,
    }))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(json.dumps({"ok": False, "error": str(e)[:200]}), file=sys.stderr)
        sys.exit(2)
