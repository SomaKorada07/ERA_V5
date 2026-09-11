"""
run_all.py — reproduce every number and figure in the README, in order.

    python3 run_all.py

Runs on CPU, deterministically, in a few minutes. Each part also runs
standalone (e.g. `python3 part1_2_adam_by_hand.py`).
"""

import runpy
import sys

PARTS = [
    ("part1_2_adam_by_hand", "Parts 1 & 2 — Adam by hand + bias correction"),
    ("part3_update_ratio",   "Part 3 — update-to-weight ratio & warmup"),
    ("part4_cosine_vs_wsd",  "Part 4 — cosine vs WSD"),
    ("part5_lr_sweep_mup",   "Part 5 — LR sweep vs width & transfer to 4096"),
]

if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for mod, title in PARTS:
        if only and only not in mod:
            continue
        print("\n" + "#" * 78)
        print("#", title)
        print("#" * 78)
        runpy.run_module(mod, run_name="__main__")
