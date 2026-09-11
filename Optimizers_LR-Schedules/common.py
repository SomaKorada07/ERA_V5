"""
common.py — shared model, data, schedules, and utilities.

Everything here is deliberately small and CPU-deterministic so the whole
assignment reproduces on a laptop in a couple of minutes with identical numbers.

Design choices that matter for the experiments:
  * A synthetic teacher/student regression task (no downloads, fully controlled
    signal-to-noise). The student is a plain MLP whose *width* is a knob — this
    is what Part 5 (muP / LR transfer) needs.
  * muP is implemented explicitly (not via a library) so the parametrization is
    auditable: per-layer LR multipliers, init scaling, and an output multiplier.
"""

import math
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cpu")  # CPU on purpose: bit-for-bit reproducible numbers.


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------------
# Synthetic teacher/student data
# -----------------------------------------------------------------------------
class Teacher:
    """A fixed random 2-layer tanh network that generates the target signal."""

    def __init__(self, in_dim=32, hidden=128, out_dim=1, seed=1234, noise=0.05):
        g = torch.Generator().manual_seed(seed)
        self.W1 = torch.randn(in_dim, hidden, generator=g) / math.sqrt(in_dim)
        self.W2 = torch.randn(hidden, out_dim, generator=g) / math.sqrt(hidden)
        self.noise = noise
        self.in_dim = in_dim

    def sample(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, self.in_dim, generator=g)
        y = torch.tanh(x @ self.W1) @ self.W2
        y = y + self.noise * torch.randn(n, y.shape[1], generator=g)
        return x, y


# -----------------------------------------------------------------------------
# Student MLP with optional muP parametrization
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    """
    3 linear layers: in_dim -> width -> width -> out_dim, GELU activations.

    parametrization:
      'sp'  = standard parametrization (PyTorch defaults, fan_in init, no LR
              multipliers). Optimal LR drifts with width.
      'mup' = maximal update parametrization. Init and a per-layer LR multiplier
              are set so the optimal LR is (approximately) width-invariant.

    For muP we follow the standard Adam recipe (Tensor Programs V / the `mup`
    package), reduced to this 3-layer MLP:

      width_mult = width / base_width
      * input layer  (in_dim -> width):  init std = 1/sqrt(in_dim),  lr_mult = 1
      * hidden layer (width  -> width):  init std = 1/sqrt(width),   lr_mult = 1/width_mult
      * output layer (width  -> out):    init = 0,                   lr_mult = 1/width_mult,
                                         and logits are multiplied by 1/width_mult.
    """

    def __init__(self, in_dim=32, width=256, out_dim=1, parametrization="sp",
                 base_width=256, seed=0):
        super().__init__()
        self.parametrization = parametrization
        self.width = width
        self.base_width = base_width
        self.width_mult = width / base_width

        self.fc1 = nn.Linear(in_dim, width, bias=True)
        self.fc2 = nn.Linear(width, width, bias=True)
        self.fc3 = nn.Linear(width, out_dim, bias=True)
        self.act = nn.GELU()

        self.output_mult = 1.0 / self.width_mult if parametrization == "mup" else 1.0

        self._init_weights(in_dim, width, out_dim, seed)

    def _init_weights(self, in_dim, width, out_dim, seed):
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            if self.parametrization == "mup":
                self.fc1.weight.normal_(0, 1.0 / math.sqrt(in_dim), generator=g)
                self.fc2.weight.normal_(0, 1.0 / math.sqrt(width), generator=g)
                self.fc3.weight.zero_()  # muP output init = 0
            else:  # standard parametrization: fan_in init on every layer
                self.fc1.weight.normal_(0, 1.0 / math.sqrt(in_dim), generator=g)
                self.fc2.weight.normal_(0, 1.0 / math.sqrt(width), generator=g)
                self.fc3.weight.normal_(0, 1.0 / math.sqrt(width), generator=g)
            self.fc1.bias.zero_()
            self.fc2.bias.zero_()
            self.fc3.bias.zero_()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        return self.fc3(h) * self.output_mult

    def param_groups_for_adam(self, base_lr):
        """
        Build Adam param groups. Under muP the hidden and output *weights* get an
        LR multiplier of 1/width_mult; biases have fan-in 1 (Theta(1)) so they
        keep the full base LR, like the input layer. Under SP every group uses
        base_lr.
        """
        if self.parametrization == "mup":
            m = 1.0 / self.width_mult
            return [
                {"params": [self.fc1.weight], "lr": base_lr},
                {"params": [self.fc2.weight], "lr": base_lr * m},
                {"params": [self.fc3.weight], "lr": base_lr * m},
                # biases (fan-in 1) stay at the full base LR under muP
                {"params": [self.fc1.bias, self.fc2.bias, self.fc3.bias],
                 "lr": base_lr},
            ]
        return [{"params": self.parameters(), "lr": base_lr}]

    def named_layers(self):
        return [("fc1", self.fc1), ("fc2", self.fc2), ("fc3", self.fc3)]


# -----------------------------------------------------------------------------
# Learning-rate schedules (return a multiplier in [0, 1] on the base LR)
# -----------------------------------------------------------------------------
def lr_cosine(step, total_steps, warmup, min_ratio=0.0):
    if step < warmup:
        return (step + 1) / warmup
    prog = (step - warmup) / max(1, (total_steps - warmup))
    prog = min(1.0, prog)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))


def lr_wsd(step, total_steps, warmup, decay_frac=0.2, min_ratio=0.0):
    """Warmup–Stable–Decay: linear warmup, flat plateau, then a short decay tail."""
    decay_steps = int(decay_frac * total_steps)
    decay_start = total_steps - decay_steps
    if step < warmup:
        return (step + 1) / warmup
    if step < decay_start:
        return 1.0
    prog = (step - decay_start) / max(1, decay_steps)
    prog = min(1.0, prog)
    return min_ratio + (1 - min_ratio) * (1 - prog)  # linear decay to min_ratio


def lr_warmup_constant(step, warmup):
    """Linear warmup then constant — used by Part 3 to isolate the warmup effect."""
    return min(1.0, (step + 1) / warmup)
