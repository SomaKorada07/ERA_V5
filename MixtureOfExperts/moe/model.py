"""A small GPT-style byte-level LM whose feed-forward block is either

  * ``dense``  — one SwiGLU feed-forward network (the "linear/dense" model), or
  * ``moe``    — a Mixture-of-Experts layer: a router picks the top-k of N SwiGLU
                 experts per token and sums their outputs with the router weights.

Everything else (attention, norms, embeddings) is identical, so the two differ only
in the feed-forward, which is exactly what Session 14 swaps. The MoE layer follows the
notes: softmax router over N experts, keep top-k and renormalise, SwiGLU experts

    E_i(x) = W_down( SiLU(W_gate x) ⊙ W_up x )

and a Switch-style load-balancing auxiliary loss

    L_aux = α · N · Σ_i f_i · P_i

with f_i the fraction of tokens routed to expert i and P_i its mean router probability.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 256           # byte-level
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False
    # feed-forward
    ffn: str = "dense"              # "dense" | "moe"
    d_ff: int = 1536               # dense hidden width (== 4 * n_embd)
    # MoE
    n_experts: int = 8
    top_k: int = 2
    expert_width: int = 768        # == d_ff // top_k  -> active width equals the dense FFN
    aux_coef: float = 0.01
    router_softmax: bool = True
    routing: str = "sparse"        # "sparse" (dispatch only chosen experts) | "masked" (compute all)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: W_down( SiLU(W_gate x) * W_up x )."""

    def __init__(self, d_in: int, d_hidden: int, bias: bool = False):
        super().__init__()
        self.w_gate = nn.Linear(d_in, d_hidden, bias=bias)
        self.w_up = nn.Linear(d_in, d_hidden, bias=bias)
        self.w_down = nn.Linear(d_hidden, d_in, bias=bias)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoE(nn.Module):
    """Mixture-of-Experts feed-forward: router + N SwiGLU experts, top-k routing."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.aux_coef = cfg.aux_coef
        self.router_softmax = cfg.router_softmax
        self.routing = cfg.routing
        self.router = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.n_embd, cfg.expert_width, bias=cfg.bias) for _ in range(cfg.n_experts)]
        )
        self.last_aux = torch.tensor(0.0)

    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(-1, C)                       # (N, C)
        N = xf.size(0)
        logits = self.router(xf)                    # (N, E)
        probs = (F.softmax(logits, dim=-1) if self.router_softmax
                 else torch.sigmoid(logits))
        topv, topi = probs.topk(self.top_k, dim=-1)  # (N, k)
        gates = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)  # renormalise to sum 1

        # full (N, E) gate matrix, nonzero only at the chosen experts
        full_gates = torch.zeros_like(probs).scatter_(1, topi, gates)

        if self.routing == "sparse":
            # Dispatch: run each expert only on the tokens routed to it (top_k of N),
            # so the compute per token matches the dense FFN. This is the real MoE.
            out = torch.zeros_like(xf)
            k = self.top_k
            flat_tok = torch.arange(N, device=xf.device).repeat_interleave(k)  # (N*k,)
            flat_exp = topi.reshape(-1)                                        # (N*k,)
            flat_gate = gates.reshape(-1)                                      # (N*k,)
            for e, expert in enumerate(self.experts):
                sel = (flat_exp == e).nonzero(as_tuple=True)[0]
                if sel.numel() == 0:
                    continue
                tok = flat_tok[sel]
                ye = expert(xf[tok])
                out.index_add_(0, tok, flat_gate[sel].unsqueeze(-1) * ye)
        else:
            # Masked: compute every expert on all tokens and combine by the (mostly-zero)
            # gates. Fixed shapes (simplest, but computes all experts).
            out = torch.zeros_like(xf)
            for e, expert in enumerate(self.experts):
                out = out + full_gates[:, e:e + 1] * expert(xf)

        # Switch-style load-balancing auxiliary loss
        dispatch = torch.zeros_like(probs).scatter_(1, topi, 1.0)  # (N,E) one-hot of picks
        f_i = dispatch.mean(dim=0)                  # fraction of tokens per expert (no grad path)
        P_i = probs.mean(dim=0)                     # mean router prob per expert (has grad)
        self.last_aux = self.aux_coef * self.n_experts * torch.sum(f_i * P_i)

        # routing stats for logging (fraction of load on the busiest expert, dead experts)
        with torch.no_grad():
            load = f_i * self.n_experts / self.top_k  # 1.0 == perfectly balanced
            self.last_maxload = load.max().item()
            self.last_dead = int((dispatch.sum(dim=0) == 0).sum().item())
        return out.reshape(B, T, C)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.is_moe = cfg.ffn == "moe"
        self.ff = MoE(cfg) if self.is_moe else SwiGLU(cfg.n_embd, cfg.d_ff, bias=cfg.bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x


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
        self.wte.weight = self.lm_head.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def active_params(self):
        """Parameters actually used per token (experts count only top_k of N)."""
        n = 0
        for name, p in self.named_parameters():
            if ".experts." in name:
                # keep only a top_k/n_experts share of the expert params
                n += int(p.numel() * self.cfg.top_k / self.cfg.n_experts)
            else:
                n += p.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = aux = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            aux = sum((blk.ff.last_aux for blk in self.blocks if blk.is_moe),
                      start=torch.zeros((), device=idx.device))
        return logits, loss, aux

    def moe_stats(self):
        """Average busiest-expert load and dead-expert count across MoE layers."""
        loads = [blk.ff.last_maxload for blk in self.blocks if blk.is_moe]
        deads = [blk.ff.last_dead for blk in self.blocks if blk.is_moe]
        if not loads:
            return None
        return {"max_load": sum(loads) / len(loads), "dead_experts": sum(deads)}

    def configure_optimizer(self, weight_decay, lr, betas):
        decay = [p for p in self.parameters() if p.dim() >= 2 and p.requires_grad]
        nodecay = [p for p in self.parameters() if p.dim() < 2 and p.requires_grad]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": nodecay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)
