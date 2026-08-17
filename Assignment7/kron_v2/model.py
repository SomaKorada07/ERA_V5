"""
Small GPT with a Kronecker input pathway and two selectable output heads:

  * "softmax"     : standard learned d_model -> |V| head (the V1 baseline output side)
  * "kron_decode" : the V2 invertible head. Predicts the next token's codec as d_p
                    parallel (d_c+1)-way byte classifiers. NO |V| in its shape, so its
                    parameter count is CONSTANT in the vocabulary size, and it can emit
                    any UTF-8 string of <= d_p bytes -- including tokens never trained on.
  * "gaussian"    : the V2 "Hypothesis B" head (paper Sec 8.5). Predicts a Gaussian
                    N(mu, diag(sigma^2)) over codec space (mu, log-var each in R^D).
                    Trained with Gaussian NLL against the target token's codec vector.
                    A single Gaussian is unimodal, so on multimodal targets it collapses
                    to the conditional MEAN (a blend) -- the honest baseline.
  * "mixture"     : the fix. A MIXTURE of `n_comp` isotropic Gaussians over codec space
                    (weights pi_m, means mu_m in R^D, scalar log-var per component).
                    Trained with mixture NLL, it places one component on each mode, so it
                    represents a genuine multimodal distribution over next tokens and can
                    be sampled into COHERENT tokens. Also O(1) in |V|.

Both arms share an identical transformer body and identical Kronecker *input*, so any
difference is attributable to the output side alone.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from codec import D_C, D_P, N_OUT_CLASSES

D = D_C * D_P    # codec dimension


class KroneckerInput(nn.Module):
    """gpu_table variant: fixed codec table [|V|, D] -> learned projection D -> d_model."""
    def __init__(self, codec_table, d_model):
        super().__init__()
        V, D = codec_table.shape
        self.register_buffer("K", torch.tensor(codec_table))   # fixed, no grad
        self.proj = nn.Linear(D, d_model, bias=False)
        nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(D))

    def forward(self, idx):                # idx: [B, T] token ids
        return self.proj(self.K[idx])      # [B, T, d_model]


class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, codec_table, vocab_size, block_size,
                 d_model=192, n_layer=4, n_head=6, dropout=0.1, head="softmax",
                 n_comp=4):
        super().__init__()
        self.block_size = block_size
        self.head_type = head
        self.n_comp = n_comp
        self.tok = KroneckerInput(codec_table, d_model)
        self.pos = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)

        if head == "softmax":
            self.head = nn.Linear(d_model, vocab_size, bias=False)
        elif head == "kron_decode":
            # d_p position-slots, each a (d_c+1)-way byte classifier.  Shape is
            # independent of vocab_size -> this is the whole point of V2.
            self.head = nn.Linear(d_model, D_P * N_OUT_CLASSES, bias=True)
        elif head == "gaussian":
            # Hypothesis B: predict mu (R^D) and log-variance (R^D) of a diagonal
            # Gaussian over codec space.  Shape is independent of vocab_size.
            self.head = nn.Linear(d_model, 2 * D, bias=True)
            # start with a modest variance so early NLL is well-behaved
            nn.init.zeros_(self.head.weight[D:])
            with torch.no_grad():
                self.head.bias[D:].fill_(-2.0)      # sigma^2 ~ exp(-2) ~ 0.135
        elif head == "mixture":
            # M components: means (M*D) | log-var scalars (M) | mixing logits (M)
            self.head = nn.Linear(d_model, n_comp * D + 2 * n_comp, bias=True)
            with torch.no_grad():
                # Break symmetry: start component means at DISTINCT points on the
                # codec unit-shell, and shrink the mean-weights so those distinct
                # biases (not random context weights) dominate early training --
                # otherwise all components collapse onto the conditional mean.
                self.head.weight[:n_comp * D] *= 0.02
                spread = torch.randn(n_comp, D)
                spread = spread / spread.norm(dim=1, keepdim=True)   # unit norm rows
                self.head.bias[:n_comp * D] = spread.reshape(-1)
                self.head.bias[n_comp * D: n_comp * D + n_comp].fill_(-1.2)  # var~0.3
        else:
            raise ValueError(head)

    def head_params(self):
        return sum(p.numel() for p in self.head.parameters())

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        x = self.drop(x)
        mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        out = self.head(x)
        if self.head_type == "kron_decode":
            out = out.view(B, T, D_P, N_OUT_CLASSES)
        return out

    # -- losses -------------------------------------------------------------- #
    def loss_softmax(self, logits, targets):          # targets: [B,T] token ids
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1))

    def loss_kron(self, logits, byte_targets):        # byte_targets: [B,T,d_p]
        B, T, P, C = logits.shape
        return F.cross_entropy(logits.reshape(-1, C), byte_targets.reshape(-1))
