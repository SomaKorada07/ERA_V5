"""
Hypothesis B on the main LM task: does the mixture-Gaussian head sample COHERENT
continuations on real (templated-grammar) text?

Setup mirrors train.py (Proof 3a): identical GPT body + Kronecker input; the corpus is
the same templated grammar, whose next token is genuinely multimodal (after "the" comes
one of many adjectives/determiners; after an adjective, one of many nouns).

We train the mixture head over full sequences, then GENERATE autoregressively by:
  1. computing the mixture over codec space at the last position,
  2. sampling a component m ~ pi,
  3. decoding component mean mu_m back to a token -- either codebook-FREE (per-column
     argmax over the 256x32 codec, no vocabulary used) or codebook-SNAP (nearest real
     codec row, a clean readout).
The categorical Hypothesis-A head is generated the same way for contrast: its
codebook-free decode samples byte positions INDEPENDENTLY, which blends modes.

Run: python3 lm_hypb.py   ->  results/lm_hypb.json
"""

import json, random, math
import numpy as np
import torch
import torch.nn.functional as F

import data
from codec import (build_codec_table, byte_targets, labels_to_string, D_C, D_P)
from model import GPT, D

torch.manual_seed(1234); np.random.seed(1234); random.seed(1234)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BLOCK = 12
N_COMP = 12
VAR = [0.5]

# ------------------------------------------------------------------ data ----- #
tokens = data.generate(n_sentences=5000, seed=0)
vocab, stoi = data.build_vocab(tokens)
V = len(vocab)
ids = np.array([stoi[t] for t in tokens], dtype=np.int64)
split = int(0.9 * len(ids)); train_ids = ids[:split]

in_table = build_codec_table(vocab, znorm=True)
K_raw = torch.tensor(build_codec_table(vocab, znorm=False)).to(DEVICE)     # [V,D] unit norm
btargets = torch.tensor(byte_targets(vocab)).to(DEVICE)
VOCAB_SET = set(vocab)
print(f"device={DEVICE}  tokens={len(ids)}  vocab={V}  block={BLOCK}  M={N_COMP}")


def get_batch(bs=48):
    ix = np.random.randint(0, len(train_ids) - BLOCK - 1, size=bs)
    x = np.stack([train_ids[i:i + BLOCK] for i in ix])
    y = np.stack([train_ids[i + 1:i + 1 + BLOCK] for i in ix])
    return torch.tensor(x).to(DEVICE), torch.tensor(y).to(DEVICE)

# --------------------------------------------------------- mixture helpers --- #
def split_mix(out):                        # out [...,M*D+2M]
    means = out[..., :N_COMP * D].view(*out.shape[:-1], N_COMP, D)
    logpi = torch.log_softmax(out[..., -N_COMP:], dim=-1)
    return means, logpi

def comp_ll(means, k):
    """log N(k; mu_m, VAR I) up to const, memory-efficiently (no [N,M,D] tensor).
    ||k-mu||^2 = ||k||^2 - 2 k.mu + ||mu||^2 ; k is unit norm so ||k||^2=1."""
    kn = (k * k).sum(-1, keepdim=True)                      # [N,1]
    mn = (means * means).sum(-1)                            # [N,M]
    cross = torch.einsum("nd,nmd->nm", k, means)           # [N,M]
    return -0.5 * (kn - 2 * cross + mn) / VAR[0]

@torch.no_grad()
def sinkhorn(scores, eps=1.0, iters=3):
    Q = torch.exp((scores - scores.max()) / eps).t()
    Q = Q / (Q.sum() + 1e-8)
    M, B = Q.shape
    for _ in range(iters):
        Q = Q / (Q.sum(1, keepdim=True) + 1e-8) / M
        Q = Q / (Q.sum(0, keepdim=True) + 1e-8) / B
    return (Q * B).t()

def mix_balanced_loss(out, k):
    means, logpi = split_mix(out)
    joint = logpi + comp_ll(means, k)
    Q = sinkhorn(joint.detach())
    return -(Q * joint).sum(1).mean()

def mix_nll(out, k):
    means, logpi = split_mix(out)
    return -(torch.logsumexp(logpi + comp_ll(means, k), dim=1)).mean()

# --------------------------------------------------------- decode helpers ---- #
def decode_mean_free(vec):
    """Codebook-FREE decode of a codec-space vector: per-column argmax over the 256x32
    grid, keeping the contiguous prefix of 'active' columns.  No vocabulary used."""
    M = vec.view(D_C, D_P)
    colmax, args = M.max(0)
    peak = colmax.max().item()
    thr = 0.35 * peak
    out = []
    for p in range(D_P):
        if colmax[p].item() <= thr:
            break
        out.append(int(args[p].item()))
    return bytes(out).decode("utf-8", errors="replace")

def snap_id(vec):
    """Codebook-SNAP: nearest real codec row (||K||=1 -> nearest = max dot)."""
    return int((vec @ K_raw.T).argmax().item())

# --------------------------------------------------------------- training ---- #
def train(head, steps):
    n_layer = 3
    model = GPT(in_table, V, BLOCK, d_model=128, n_layer=n_layer, n_head=4,
                dropout=0.0, head=head, n_comp=N_COMP).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        x, y = get_batch()
        out = model(x)
        if head == "mixture":
            VAR[0] = 2.0 * (0.02 / 2.0) ** (step / steps)   # deterministic annealing
            flat_out = out.reshape(-1, out.size(-1))
            flat_k = K_raw[y.reshape(-1)]
            loss = mix_balanced_loss(flat_out, flat_k)
        else:                                                # kron_decode (Hyp A)
            loss = F.cross_entropy(out.reshape(-1, out.size(-1)),
                                   btargets[y].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"  [{head:11}] step {step:4d}  loss {loss.item():.4f}"
                  + (f"  var {VAR[0]:.3f}" if head == 'mixture' else ""))
    return model

# --------------------------------------------------------------- generate ---- #
@torch.no_grad()
def gen_mixture(model, prompt_words, n=10, decode_var=0.15):
    VAR[0] = decode_var
    ids_ = [stoi[w] for w in prompt_words]
    snap_words, free_words = [], []
    for _ in range(n):
        x = torch.tensor([ids_[-BLOCK:]]).to(DEVICE)
        out = model(x)[0, -1]
        means, logpi = split_mix(out)
        m = int(torch.multinomial(torch.exp(logpi), 1).item())
        mu = means[m]
        sid = snap_id(mu)
        snap_words.append(vocab[sid])
        free_words.append(decode_mean_free(mu))
        ids_.append(sid)                                    # roll out on the clean token
    return snap_words, free_words

@torch.no_grad()
def gen_hyp_a(model, prompt_words, n=10, temp=0.8):
    ids_ = [stoi[w] for w in prompt_words]
    snap_words, free_words = [], []
    for _ in range(n):
        x = torch.tensor([ids_[-BLOCK:]]).to(DEVICE)
        out = model(x)[0, -1].view(D_P, -1)                 # [d_p, C]
        probs = torch.softmax(out / temp, dim=-1)
        # codebook-free: sample each byte position INDEPENDENTLY
        samp = torch.multinomial(probs, 1).squeeze(-1)       # [d_p]
        free_words.append(labels_to_string(samp.tolist()))
        # codebook-restricted: score every vocab token by summed byte log-probs
        lp = torch.log_softmax(out, dim=-1)
        score = torch.stack([lp[torch.arange(D_P), btargets[v]].sum() for v in range(V)])
        sid = int(torch.multinomial(torch.softmax(score, 0), 1).item())
        snap_words.append(vocab[sid])
        ids_.append(sid)
    return snap_words, free_words

def validity(words):
    return float(np.mean([w in VOCAB_SET for w in words]))

@torch.no_grad()
def components_for(model, prompt_words, topk=6, decode_var=0.15):
    VAR[0] = decode_var
    x = torch.tensor([[stoi[w] for w in prompt_words][-BLOCK:]]).to(DEVICE)
    out = model(x)[0, -1]
    means, logpi = split_mix(out)
    pi = torch.exp(logpi)
    order = torch.argsort(pi, descending=True)[:topk]
    return [{"word": vocab[snap_id(means[m])],
             "free": decode_mean_free(means[m]),
             "pi": round(pi[m].item(), 3)} for m in order.tolist()]


if __name__ == "__main__":
    print("== training Hypothesis A (kron_decode) on the LM corpus ==")
    m_a = train("kron_decode", steps=1500)
    print("== training Hypothesis B+ (mixture) on the LM corpus ==")
    m_b = train("mixture", steps=3000)

    prompts = [["the"], ["the", "quick"], ["a", "silent", "fox"],
               ["the", "ancient", "river"]]
    rng_prompts = [["the", "quick", "fox"], ["a", "clever", "scholar", "builds"]]

    # generation samples
    gens = []
    for p in [["the", "quick", "fox"], ["a", "silent", "river"],
              ["the", "ancient", "kingdom"]]:
        s_mix, f_mix = gen_mixture(m_b, p, n=10)
        s_a, f_a = gen_hyp_a(m_a, p, n=10)
        gens.append({
            "prompt": " ".join(p),
            "mixture_snap": " ".join(s_mix),
            "mixture_free": " ".join(f_mix),
            "hypA_codebook": " ".join(s_a),
            "hypA_free": " ".join(f_a),
        })
        print(f"\nprompt: {' '.join(p)}")
        print("  mixture (codebook-free):", " ".join(f_mix))
        print("  mixture (codebook-snap):", " ".join(s_mix))
        print("  Hyp A  (codebook-free ):", " ".join(f_a))

    # aggregate codebook-free validity over many rollouts
    mix_valid, a_valid = [], []
    for _ in range(20):
        p = random.choice([["the"], ["a"], ["the", "quick"], ["a", "silent"]])
        _, fm = gen_mixture(m_b, p, n=8)
        _, fa = gen_hyp_a(m_a, p, n=8)
        mix_valid.append(validity(fm)); a_valid.append(validity(fa))
    mix_valid = round(float(np.mean(mix_valid)), 4)
    a_valid = round(float(np.mean(a_valid)), 4)

    # per-context component analysis
    comp_analysis = [{"prompt": " ".join(p), "components": components_for(m_b, p)}
                     for p in prompts]
    for c in comp_analysis:
        print(f"\nafter '{c['prompt']}'  ->  " +
              ", ".join(f"{d['word']}({d['pi']})" for d in c["components"]))

    result = {
        "device": DEVICE, "vocab": V, "block": BLOCK, "n_components": N_COMP,
        "codebook_free_validity": {"mixture": mix_valid, "hyp_a_independent": a_valid},
        "generations": gens,
        "component_analysis": comp_analysis,
        "note": "Mixture samples a component then decodes it in codec space -> coherent "
                "words even codebook-free. Hyp A's codebook-free decode samples byte "
                "positions independently -> blends. Both are O(1) in |V|.",
    }
    with open("results/lm_hypb.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nCODEBOOK-FREE token validity:  mixture", mix_valid,
          " vs  Hyp A (independent)", a_valid)
