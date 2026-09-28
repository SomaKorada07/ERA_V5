"""Training loop, throughput + peak-memory measurement, and a max-batch finder."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List

import torch

from .data import TokenData
from .model import GPT, GPTConfig


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _reset_peak(device: str):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def _current_mem(device: str) -> int:
    """High-water proxy sampled per step.

    On MPS ``current_allocated_memory`` only reflects live tensors *between* steps and
    misses the transient forward/backward spike (the thing that actually OOMs), so we
    use ``driver_allocated_memory`` — the total the Metal driver holds for the process,
    which climbs to and holds the high-water mark.
    """
    if device == "cuda":
        return torch.cuda.memory_allocated()
    if device == "mps":
        return torch.mps.driver_allocated_memory()
    return 0


def _peak_mem(device: str, running_peak: int) -> int:
    """CUDA reports its own peak; MPS/CPU rely on the polled running peak."""
    if device == "cuda":
        return max(running_peak, torch.cuda.max_memory_allocated())
    return running_peak


def _sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@dataclass
class TrainResult:
    name: str
    mode: str
    integrator: str
    batch_size: int
    block_size: int
    tokens_per_step: int
    steps: int
    tokens_seen: int
    final_train_loss: float
    final_val_loss: float
    tokens_per_sec: float
    peak_mem_bytes: int
    wall_time_s: float
    params: int
    step_size: float
    loss_history: List[dict] = field(default_factory=list)
    note: str = ""

    def summary(self) -> str:
        return (
            f"{self.name}: params={self.params/1e6:.2f}M  bs={self.batch_size}  "
            f"tok/step={self.tokens_per_step:,}  final_val_loss={self.final_val_loss:.4f}  "
            f"speed={self.tokens_per_sec:,.0f} tok/s  peak_mem={self.peak_mem_bytes/1e9:.3f} GB"
        )


@torch.no_grad()
def evaluate(model, val: TokenData, batch_size, block_size, device, iters=40):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = val.get_batch(batch_size, block_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def _lr_at(step, total, lr, warmup, min_ratio=0.1):
    if step < warmup:
        return lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def train_run(
    cfg: GPTConfig,
    name: str,
    batch_size: int,
    token_budget: int,
    train_bin: str,
    val_bin: str,
    device: str,
    lr: float = 6e-4,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.1,
    log_every: int = 50,
    eval_every: int = 400,
    grad_clip: float = 1.0,
    max_steps: int | None = None,
) -> TrainResult:
    """Train until ``token_budget`` tokens have been consumed (or ``max_steps``)."""
    torch.manual_seed(1337)
    train = TokenData(train_bin)
    val = TokenData(val_bin)

    model = GPT(cfg).to(device)
    model.train()
    opt = model.configure_optimizer(weight_decay, lr, (0.9, 0.95))

    tokens_per_step = batch_size * cfg.block_size
    steps = token_budget // tokens_per_step
    if max_steps is not None:
        steps = min(steps, max_steps)
    warmup = max(10, int(steps * warmup_ratio))

    _reset_peak(device)
    running_peak = 0
    loss_history = []
    _sync(device)
    t0 = time.time()
    last_loss = float("nan")

    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, steps, lr, warmup)
        x, y = train.get_batch(batch_size, cfg.block_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        running_peak = max(running_peak, _current_mem(device))
        last_loss = loss.item()

        if step % log_every == 0 or step == steps - 1:
            elapsed = time.time() - t0
            tps = (step + 1) * tokens_per_step / max(elapsed, 1e-9)
            loss_history.append({"step": step, "tokens": (step + 1) * tokens_per_step,
                                 "train_loss": last_loss, "tok_per_s": tps})
            print(f"[{name}] step {step:5d}/{steps}  loss {last_loss:.4f}  "
                  f"{tps:,.0f} tok/s  peak {running_peak/1e9:.2f}GB", flush=True)

    _sync(device)
    wall = time.time() - t0
    peak = _peak_mem(device, running_peak)
    final_val = evaluate(model, val, batch_size, cfg.block_size, device)
    tokens_seen = steps * tokens_per_step
    tps = tokens_seen / max(wall, 1e-9)

    res = TrainResult(
        name=name, mode=cfg.mode, integrator=cfg.integrator, batch_size=batch_size,
        block_size=cfg.block_size, tokens_per_step=tokens_per_step, steps=steps,
        tokens_seen=tokens_seen, final_train_loss=last_loss, final_val_loss=final_val,
        tokens_per_sec=tps, peak_mem_bytes=peak, wall_time_s=wall,
        params=model.num_params(), step_size=cfg.step_size, loss_history=loss_history,
    )
    del model, opt
    _reset_peak(device)
    return res


def measure_step_memory(cfg: GPTConfig, batch_size: int, device: str, warmup=3, iters=8):
    """One-config micro-measurement: peak bytes and tokens/sec for train steps.

    Returns (peak_bytes, tokens_per_sec) or raises on OOM.
    """
    torch.manual_seed(0)
    model = GPT(cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    T = cfg.block_size
    _reset_peak(device)
    running_peak = 0

    def one():
        x = torch.randint(0, cfg.vocab_size, (batch_size, T), device=device)
        y = torch.randint(0, cfg.vocab_size, (batch_size, T), device=device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(warmup):
        one()
    _sync(device)
    t0 = time.time()
    for _ in range(iters):
        one()
        running_peak = max(running_peak, _current_mem(device))
    _sync(device)
    dt = time.time() - t0
    peak = _peak_mem(device, running_peak)
    tps = iters * batch_size * T / dt
    del model, opt
    _reset_peak(device)
    return peak, tps


def find_max_batch(cfg: GPTConfig, device: str, start=8, cap=4096):
    """Double the batch until OOM (or cap); return the largest that ran + its stats."""
    last_ok = None
    bs = start
    while bs <= cap:
        try:
            peak, tps = measure_step_memory(cfg, bs, device)
            last_ok = {"batch_size": bs, "peak_mem_bytes": peak, "tokens_per_sec": tps}
            print(f"[maxbatch:{cfg.mode}] bs={bs} OK  peak={peak/1e9:.2f}GB  {tps:,.0f} tok/s", flush=True)
            bs *= 2
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "MPS" in str(e):
                print(f"[maxbatch:{cfg.mode}] bs={bs} OOM", flush=True)
                break
            raise
    return last_ok
