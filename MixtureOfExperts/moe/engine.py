"""Training loop with loss/throughput logging and MoE routing stats."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List

import torch

from .data import ByteData
from .model import GPT, GPTConfig


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


@dataclass
class RunResult:
    name: str
    ffn: str
    total_params: int
    active_params: int
    batch_size: int
    block_size: int
    steps: int
    tokens_seen: int
    final_train_loss: float
    final_val_loss: float
    tokens_per_sec: float
    wall_time_s: float
    history: List[dict] = field(default_factory=list)
    note: str = ""

    def summary(self):
        return (f"{self.name}: ffn={self.ffn} params={self.total_params/1e6:.2f}M "
                f"(active {self.active_params/1e6:.2f}M) val_loss={self.final_val_loss:.4f} "
                f"speed={self.tokens_per_sec:,.0f} tok/s")


@torch.no_grad()
def evaluate(model, val: ByteData, batch_size, block_size, device, iters=50):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = val.get_batch(batch_size, block_size, device)
        _, loss, _ = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def _lr_at(step, total, lr, warmup, min_ratio=0.1):
    if step < warmup:
        return lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def train_run(cfg: GPTConfig, name, batch_size, token_budget, train_bin, val_bin, device,
              lr=3e-4, warmup_ratio=0.03, weight_decay=0.1, grad_clip=1.0,
              log_every=25, eval_every=250, note="", init_from: GPT | None = None):
    torch.manual_seed(1337)
    train, val = ByteData(train_bin), ByteData(val_bin)
    model = GPT(cfg).to(device)
    if init_from is not None:
        _upcycle(model, init_from)
    model.train()
    opt = model.configure_optimizer(weight_decay, lr, (0.9, 0.95))

    tokens_per_step = batch_size * cfg.block_size
    steps = token_budget // tokens_per_step
    warmup = max(10, int(steps * warmup_ratio))

    history = []
    _sync(device)
    t0 = time.time()
    last_loss = float("nan")
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, steps, lr, warmup)
        x, y = train.get_batch(batch_size, cfg.block_size, device)
        _, loss, aux = model(x, y)
        total = loss if aux is None else loss + aux
        opt.zero_grad(set_to_none=True)
        total.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        last_loss = loss.item()

        if step % log_every == 0 or step == steps - 1:
            elapsed = time.time() - t0
            tps = (step + 1) * tokens_per_step / max(elapsed, 1e-9)
            rec = {"step": step, "tokens": (step + 1) * tokens_per_step,
                   "train_loss": last_loss, "tok_per_s": tps}
            st = model.moe_stats()
            if st:
                rec.update(aux=float(aux.item()), max_load=st["max_load"], dead=st["dead_experts"])
            history.append(rec)
            extra = (f" aux {rec['aux']:.3f} maxload {rec['max_load']:.2f} dead {rec['dead']}"
                     if st else "")
            print(f"[{name}] step {step:5d}/{steps} loss {last_loss:.4f} {tps:,.0f} tok/s{extra}",
                  flush=True)

    _sync(device)
    wall = time.time() - t0
    final_val = evaluate(model, val, batch_size, cfg.block_size, device)
    tokens_seen = steps * tokens_per_step
    res = RunResult(
        name=name, ffn=cfg.ffn, total_params=model.num_params(),
        active_params=model.active_params(), batch_size=batch_size, block_size=cfg.block_size,
        steps=steps, tokens_seen=tokens_seen, final_train_loss=last_loss,
        final_val_loss=final_val, tokens_per_sec=tokens_seen / max(wall, 1e-9),
        wall_time_s=wall, history=history, note=note,
    )
    return res, model


def _upcycle(moe_model: GPT, dense_model: GPT):
    """Initialise an MoE model from a trained dense model: copy shared weights, and copy
    each dense SwiGLU FFN into every expert (Expert-Upcycling). Requires expert_width to
    equal the dense d_ff. Router is left at its small random init.
    """
    dense_sd = dense_model.state_dict()
    with torch.no_grad():
        for name, p in moe_model.named_parameters():
            if name in dense_sd and dense_sd[name].shape == p.shape:
                p.copy_(dense_sd[name])
        # copy dense FFN -> each expert
        for li, blk in enumerate(moe_model.blocks):
            if not blk.is_moe:
                continue
            for w in ("w_gate", "w_up", "w_down"):
                src = dense_sd[f"blocks.{li}.ff.{w}.weight"]
                for expert in blk.ff.experts:
                    getattr(expert, w).weight.copy_(src)
