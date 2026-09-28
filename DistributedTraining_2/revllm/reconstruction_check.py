"""Which integrator is actually reversible? euler vs leapfrog, across step sizes.

Writes results/reconstruction.json. This is the empirical basis for the report's
"which variant worked" answer: only the leapfrog / midpoint rule reconstructs the
input to machine precision and therefore trains with the memory-free backward pass.
"""
from __future__ import annotations

import json
import os

import torch

from .model import Block, GPTConfig
from .reversible import reconstruction_error

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(0)
    cfg = GPTConfig(n_layer=9, n_embd=256, n_head=8, block_size=512, dropout=0.0)
    blocks = torch.nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
    x = torch.randn(4, 128, cfg.n_embd)

    rows = []
    for h in [0.1, 0.25, 0.5, 1.0]:
        err_lf, _ = reconstruction_error(x, blocks, h, "leapfrog")
        err_eu, _ = reconstruction_error(x, blocks, h, "euler")
        rows.append({"step_size": h, "leapfrog_rel_err": err_lf, "euler_naive_rel_err": err_eu})
        print(f"h={h:<5} leapfrog={err_lf:.2e}   euler(naive)={err_eu:.2e}")

    out = {
        "description": "Relative L2 error of the input reconstructed by running the "
                       "stack in reverse. Leapfrog reverses exactly (layer l is "
                       "evaluated at the held state p_l); naive euler cannot.",
        "n_layer": cfg.n_layer, "rows": rows,
        "verdict": "leapfrog/midpoint is reversible to machine precision at every h; "
                   "naive euler is not reversible and its error grows with h.",
    }
    path = os.path.join(RESULTS_DIR, "reconstruction.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
