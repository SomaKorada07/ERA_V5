"""A ~20M-parameter GPT-style decoder LM with a switchable layer stack.

The residual stack can run in three modes:

  * ``standard`` — ordinary pre-norm residual blocks (the Euler / h=1 update
    ``p_{l+1} = p_l + f(p_l)``). Activations are stored by autograd. This is the
    memory-hungry baseline.

  * ``reversible`` — the leapfrog / midpoint rule of Gal et al. (Nov 2025),
    ``p_{l+1} = p_{l-1} + 2h * f(p_l)``. The backward pass rebuilds every layer's
    activations by running the rule in reverse, so activation memory does not grow
    with depth. See :mod:`revllm.reversible`.

The block function ``f_theta`` is exactly "attention followed by the feed-forward
network" as a single deterministic function of the residual stream, so the same
block definition is used by both modes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .reversible import reversible_stack


@dataclass
class GPTConfig:
    vocab_size: int = 50257          # GPT-2 BPE (tiktoken "gpt2")
    block_size: int = 512            # context length
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    mlp_ratio: int = 4
    dropout: float = 0.0             # reversibility requires a deterministic forward
    bias: bool = True
    # stack behaviour
    mode: str = "standard"           # "standard" | "reversible"
    integrator: str = "euler"        # "euler" (standard) | "leapfrog"/"midpoint" (reversible)
    step_size: float = 0.5           # h in the leapfrog rule; 2h scales the block output
    # Memory-efficient LM head: compute logits + cross-entropy in row chunks, each
    # chunk gradient-checkpointed so the full [B*T, vocab] logits tensor is never
    # stored (and never built as one matmul — MPS caps single-kernel matmul size).
    loss_chunk: int = 16384          # rows (tokens) per chunk; <=0 disables chunking


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # scaled dot-product attention, causal
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.n_embd
        self.c_fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    """One transformer block exposed as a pure function ``f_theta(p)``.

    ``f_theta(p) = attn(ln1(p)) + mlp(ln2(p + attn(ln1(p))))`` — the full block
    (attention then FFN) written as a single update to the residual stream, with
    no residual add of its own. The stack (standard or reversible) is responsible
    for combining this delta with the stream.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        a = p + self.attn(self.ln_1(p))
        return (a - p) + self.mlp(self.ln_2(a))  # == attn(...) + mlp(...)


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()  # wte is tied to lm_head, counted once
        return n

    def _run_stack(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.mode == "standard":
            for blk in self.blocks:
                x = x + blk(x)          # Euler / standard residual, h absorbed = 1
            return x
        elif self.cfg.mode == "reversible":
            return reversible_stack(x, self.blocks, self.cfg.step_size)
        raise ValueError(f"unknown mode {self.cfg.mode!r}")

    def _chunked_loss(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy without ever materialising the full [B*T, vocab] logits.

        Rows are processed in chunks; each chunk's head + CE is gradient-checkpointed,
        so forward keeps at most one chunk of logits alive and backward recomputes the
        rest. This both fits MPS's single-matmul size cap and keeps head memory ~flat,
        letting the transformer-stack activations be what decides the batch ceiling.
        """
        C = x.size(-1)
        xf = x.reshape(-1, C)
        tf = targets.reshape(-1)
        n = xf.size(0)
        chunk = self.cfg.loss_chunk if self.cfg.loss_chunk > 0 else n
        head = self.lm_head

        def head_ce(xc, tc):
            return F.cross_entropy(head(xc), tc, ignore_index=-1, reduction="sum")

        total = xf.new_zeros(())
        for i in range(0, n, chunk):
            xc, tc = xf[i:i + chunk], tf[i:i + chunk]
            if self.training and xc.requires_grad:
                total = total + torch.utils.checkpoint.checkpoint(head_ce, xc, tc, use_reentrant=False)
            else:
                total = total + head_ce(xc, tc)
        return total / n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        x = self._run_stack(x)
        x = self.ln_f(x)
        if targets is None:
            return self.lm_head(x), None
        loss = self._chunked_loss(x, targets)
        return None, loss

    def configure_optimizer(self, weight_decay, lr, betas):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)
