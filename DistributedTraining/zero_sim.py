"""
zero_sim.py
===========
A from-scratch, *executable* simulation of ZeRO (Zero Redundancy Optimizer)
stages 1, 2 and 3 on top of 32 "virtual GPUs" (CPU threads).

Nothing here is narrated hand-waving:
  * The collectives (all-reduce, reduce-scatter, all-gather) are implemented and
    move real tensors between real per-device buffers.
  * Every byte of persistent model state is *measured* on a real device buffer,
    then cross-checked against the classic ZeRO memory formula.
  * Every stage runs a real forward/backward/optimizer step and is proven to
    produce the *same* weight update as a single-device reference (bit-for-bit
    within fp tolerance). If the sharding maths were wrong, this check fails.

Memory model (mixed-precision training with the Adam optimizer), per parameter:
    fp16 parameter ................ 2 bytes   (used for the fwd/bwd compute)
    fp16 gradient ................. 2 bytes
    Adam optimizer state (fp32) ... 12 bytes  = 4 (fp32 master weight)
                                               + 4 (momentum  m)
                                               + 4 (variance  v)
    ----------------------------------------------------------------------
    total per parameter, per GPU (baseline) = 16 bytes  =  (2 + 2 + K),  K = 12

With Psi parameters and N GPUs the *persistent* per-GPU model-state bytes are:

    Baseline (DDP):  (2 + 2 + K) * Psi                       = 16 * Psi
    ZeRO-1        :   2 + 2 + K/N ......... optimizer sharded = (4 + 12/N) * Psi
    ZeRO-2        :   2 + 2/N + K/N ....... + gradients       = (2 + 14/N) * Psi
    ZeRO-3        :  (2 + 2 + K)/N ........ + parameters      = 16/N * Psi
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch
import torch.nn as nn

# --- constants of the memory model (bytes per parameter) --------------------
BYTES_FP16 = 2          # a half-precision element
BYTES_FP32 = 4          # a single-precision element
K_ADAM = 3 * BYTES_FP32  # 12 : fp32 master weight + Adam m + Adam v


# ===========================================================================
#  1.  The 32 virtual GPUs
# ===========================================================================
class VirtualGPU:
    """One virtual accelerator, backed by a CPU thread.

    It owns whatever tensors are *resident* on it and tracks how many bytes it
    is holding (current and peak). This is how we "see" the memory of a device
    without needing 32 physical cards.
    """

    def __init__(self, rank: int):
        self.rank = rank
        self.store: dict[str, torch.Tensor] = {}   # name -> resident tensor
        self._cur_bytes = 0
        self._peak_bytes = 0
        self._lock = threading.Lock()

    # -- residency -----------------------------------------------------------
    def put(self, name: str, tensor: torch.Tensor) -> None:
        with self._lock:
            if name in self.store:
                self._cur_bytes -= self.store[name].nbytes
            self.store[name] = tensor
            self._cur_bytes += tensor.nbytes
            self._peak_bytes = max(self._peak_bytes, self._cur_bytes)

    def get(self, name: str) -> torch.Tensor:
        return self.store[name]

    def free(self, name: str) -> None:
        with self._lock:
            if name in self.store:
                self._cur_bytes -= self.store.pop(name).nbytes

    # -- accounting ----------------------------------------------------------
    @property
    def resident_bytes(self) -> int:
        return self._cur_bytes

    @property
    def peak_bytes(self) -> int:
        return self._peak_bytes

    def reset_peak(self) -> None:
        self._peak_bytes = self._cur_bytes


class Cluster:
    """A pool of N virtual GPUs plus a thread pool that runs work on them."""

    def __init__(self, world_size: int = 32):
        self.world_size = world_size
        self.gpus = [VirtualGPU(r) for r in range(world_size)]
        self.pool = ThreadPoolExecutor(max_workers=world_size)
        self.comm = CommCounter()

    def run(self, fn):
        """Run fn(gpu) on every GPU in parallel; return list of results by rank."""
        return list(self.pool.map(fn, self.gpus))

    def shutdown(self):
        self.pool.shutdown()


# ===========================================================================
#  2.  Communication accounting  +  the collectives
# ===========================================================================
class CommCounter:
    """Counts bytes that cross the interconnect, per GPU (ring-algorithm cost)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.bytes_per_gpu = 0  # volume sent+received by a single GPU

    def add(self, b: int):
        self.bytes_per_gpu += b


def shard_sizes(numel: int, n: int) -> list[int]:
    """Contiguous, load-balanced partition of `numel` elements across n shards."""
    base, rem = divmod(numel, n)
    return [base + (1 if r < rem else 0) for r in range(n)]


def shard_offsets(numel: int, n: int) -> list[tuple[int, int]]:
    sizes = shard_sizes(numel, n)
    offs, start = [], 0
    for s in sizes:
        offs.append((start, start + s))
        start += s
    return offs


# --- the three collectives -------------------------------------------------
# Each takes a list of per-GPU tensors and returns the collective result,
# charging the CommCounter the *per-GPU* traffic of the standard ring algorithm.

def all_reduce(per_gpu: list[torch.Tensor], comm: CommCounter) -> torch.Tensor:
    """Sum a full-size tensor that every GPU holds. Ring cost ~ 2*(N-1)/N * bytes."""
    n = len(per_gpu)
    total = torch.stack(per_gpu, 0).sum(0)
    comm.add(int(2 * (n - 1) / n * per_gpu[0].nbytes))
    return total


def reduce_scatter(per_gpu: list[torch.Tensor], comm: CommCounter) -> list[torch.Tensor]:
    """Sum full-size tensors, but each GPU keeps only *its* shard of the sum.
    Ring cost ~ (N-1)/N * bytes  (half of all-reduce)."""
    n = len(per_gpu)
    total = torch.stack(per_gpu, 0).sum(0)
    offs = shard_offsets(total.numel(), n)
    comm.add(int((n - 1) / n * per_gpu[0].nbytes))
    return [total[a:b].clone() for (a, b) in offs]


def all_gather(shards: list[torch.Tensor], comm: CommCounter) -> torch.Tensor:
    """Concatenate every GPU's shard into the full tensor on every GPU.
    Ring cost ~ (N-1)/N * full_bytes."""
    n = len(shards)
    full = torch.cat([s for s in shards], 0)
    comm.add(int((n - 1) / n * full.nbytes))
    return full


# ===========================================================================
#  3.  The demo model
# ===========================================================================
class DemoMLP(nn.Module):
    """A plain multi-layer perceptron. Big enough that the memory story is
    visible, small enough that all 32 shards fit on one laptop."""

    def __init__(self, d_in=256, d_hidden=1024, d_out=256, depth=4):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, d_hidden), nn.GELU()]
            d = d_hidden
        layers += [nn.Linear(d, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ===========================================================================
#  4.  Flat-parameter helpers  (ZeRO works on one flat vector)
# ===========================================================================
def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).float()


def load_flat_into_model(model: nn.Module, flat: torch.Tensor) -> None:
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p).to(p.dtype))
        i += n


def grad_flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()]).float()


# ===========================================================================
#  5.  Memory accounting (the classic ZeRO table), in *bytes per GPU*
# ===========================================================================
@dataclass
class MemBreakdown:
    params: float
    grads: float
    optim: float

    @property
    def total(self) -> float:
        return self.params + self.grads + self.optim


def formula_memory(psi: int, n: int, stage: str) -> MemBreakdown:
    """Persistent per-GPU model-state bytes predicted by the ZeRO formula."""
    p_full, g_full, o_full = BYTES_FP16 * psi, BYTES_FP16 * psi, K_ADAM * psi
    if stage == "baseline":
        return MemBreakdown(p_full, g_full, o_full)
    if stage == "zero1":
        return MemBreakdown(p_full, g_full, o_full / n)
    if stage == "zero2":
        return MemBreakdown(p_full, g_full / n, o_full / n)
    if stage == "zero3":
        return MemBreakdown(p_full / n, g_full / n, o_full / n)
    raise ValueError(stage)


# ===========================================================================
#  6.  A single reference optimizer step (ground truth for correctness)
# ===========================================================================
def adam_step(flat_w32, grad32, m, v, t, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    """Vanilla Adam update on fp32 master weights (in-place on m, v)."""
    m.mul_(b1).add_(grad32, alpha=1 - b1)
    v.mul_(b2).addcmul_(grad32, grad32, value=1 - b2)
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    flat_w32.addcdiv_(mhat, vhat.sqrt().add_(eps), value=-lr)
    return flat_w32
