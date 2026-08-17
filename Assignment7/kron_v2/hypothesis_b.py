"""
Hypothesis B (V1 paper Sec 8.5): a Gaussian-over-codec output head, and the proof
that it fixes the multimodal failure of Hypothesis A.

THE PROBLEM (Hypothesis A / kron_decode).
The invertible head predicts the next token's bytes as INDEPENDENT per-position
softmaxes.  When the true next token is ambiguous -- e.g. a context legitimately
followed by "river" OR "zebra" OR "cloud" -- a point decode (per-position argmax)
mixes them into an invalid blend ("ziver"...).  That is why `free_byte_acc` was 0
on ambiguous language-model targets.

THE FIX (Hypothesis B / gaussian).
Stop predicting a point.  The head predicts a Gaussian N(mu, diag(sigma^2)) over
codec space and is trained by Gaussian NLL against the target token's codec vector.
Now the model represents a DISTRIBUTION over next tokens:
  (1) it can say HOW uncertain it is (sigma grows on ambiguous contexts);
  (2) its induced next-token distribution matches the true multimodal distribution;
  (3) SAMPLING from it + snapping to the codebook recovers the individual modes with
      the right frequencies, instead of averaging them into a blend.

Controlled task: 18 contexts.  12 are MULTIMODAL (each followed uniformly by 3 fixed
5-letter words); 6 are DETERMINISTIC (followed by exactly 1 word).  Ground-truth
next-token distributions are therefore known exactly, so KL and calibration are exact.

Run: python3 hypothesis_b.py   ->  results/hypothesis_b.json
"""

import json, random, string, math
import numpy as np
import torch
import torch.nn.functional as F

from codec import (build_codec_table, byte_targets, labels_to_string, D_C, D_P)
from model import GPT, D

torch.manual_seed(21); np.random.seed(21)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
WLEN = 5

# ----------------------------------------------------------------- task ------ #
def rand_words(n, rng, length=WLEN, exclude=set()):
    out = []
    while len(out) < n:
        w = "".join(rng.choice(string.ascii_lowercase) for _ in range(length))
        if w not in exclude and w not in out:
            out.append(w)
    return out

rng = random.Random(5)
N_MULTI, N_DET, K = 12, 6, 3
used = set()
contexts = []                        # list of (ctx_token, [continuations], is_multi)
for i in range(N_MULTI):
    conts = rand_words(K, rng, exclude=used); used.update(conts)
    contexts.append((f"ctx{i:02d}", conts, True))
for i in range(N_DET):
    conts = rand_words(1, rng, exclude=used); used.update(conts)
    contexts.append((f"dtx{i:02d}", conts, False))

vocab = [c[0] for c in contexts] + sorted(used)
stoi = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

# input codec (z-normalised, as V1); target codec (raw unit-norm kappa)
in_table = build_codec_table(vocab, znorm=True)
K_raw = torch.tensor(build_codec_table(vocab, znorm=False)).to(DEVICE)     # [V,D]
K2 = (K_raw * K_raw)                                                        # [V,D]
btargets = torch.tensor(byte_targets(vocab)).to(DEVICE)                     # [V,d_p]

# ground-truth next-token distribution per context  [n_ctx, V]
ctx_ids = torch.tensor([stoi[c[0]] for c in contexts]).to(DEVICE)
true_dist = torch.zeros(len(contexts), V, device=DEVICE)
is_multi = torch.tensor([c[2] for c in contexts], device=DEVICE)
for ci, (_, conts, _) in enumerate(contexts):
    for w in conts:
        true_dist[ci, stoi[w]] = 1.0 / len(conts)

print(f"device={DEVICE}  vocab={V}  multimodal_ctx={N_MULTI}  deterministic_ctx={N_DET}")

# training pairs (ctx -> sampled continuation)
def make_batch(bs=256):
    ci = np.random.randint(0, len(contexts), size=bs)
    x = np.array([stoi[contexts[c][0]] for c in ci], dtype=np.int64)[:, None]  # [B,1]
    y = np.array([stoi[random.choice(contexts[c][1])] for c in ci], dtype=np.int64)
    return torch.tensor(x).to(DEVICE), torch.tensor(y).to(DEVICE)

# ------------------------------------------------------- gaussian helpers ---- #
def split_gauss(out):                      # out [.,2D] -> mu, logvar  (last position)
    mu, logvar = out[..., :D], out[..., D:]
    return mu, logvar.clamp(-6.0, 4.0)

def gauss_nll(mu, logvar, k):              # per-sample, drop const
    return 0.5 * (((k - mu) ** 2) * torch.exp(-logvar) + logvar).sum(-1).mean()

def gauss_induced(mu, logvar):
    """Model's next-token distribution over the vocab: softmax over Gaussian
    log-likelihood of each codebook vector.  [N,V]  (v-constant terms dropped)."""
    P = torch.exp(-logvar)                 # 1/sigma^2  [N,D]
    score = -0.5 * (P @ K2.T) + (mu * P) @ K_raw.T          # [N,V]
    return torch.softmax(score, dim=1), score

# ------------------------------------------------------- mixture helpers ---- #
N_COMP = 4              # more components than modes (K=3): each mode can claim one
DECODE_VAR = 0.15       # variance used at DECODE time (a temperature). Training anneals
                        # sharp to force per-mode specialisation; decoding uses a
                        # moderate variance so the induced distribution reflects the modes.
VAR = [0.5]              # current isotropic variance (annealed during training).
                        # Uncertainty is carried by the mixing weights pi, not sigma;
                        # a fixed/known variance removes the D*log(sigma^2) term that
                        # made a learned-variance mixture numerically pathological.

def split_mix(out):
    """out [N, M*D+2M] -> means [N,M,D], logpi [N,M].  (variance slots unused)."""
    N = out.shape[0]
    means = out[:, :N_COMP * D].view(N, N_COMP, D)
    logpi = torch.log_softmax(out[:, -N_COMP:], dim=1)
    return means, logpi

def mix_comp_ll(means, k):
    """log N(k; mu_m, VAR * I) per component, up to a shared constant.  [N,M]."""
    diff2 = ((k[:, None, :] - means) ** 2).sum(-1)          # [N,M]
    return -0.5 * diff2 / VAR[0]

def mix_nll(out, k):
    """Soft mixture NLL (for reporting)."""
    means, logpi = split_mix(out)
    joint = logpi + mix_comp_ll(means, k)                   # [N,M]
    return -(torch.logsumexp(joint, dim=1)).mean(), joint

@torch.no_grad()
def sinkhorn(scores, eps=1.0, iters=3):
    """Balanced soft assignment (SwAV-style): every component gets ~equal batch mass,
    so no component dies -> the K modes get covered instead of one grabbing all."""
    Q = torch.exp((scores - scores.max()) / eps).t()       # [M,B]
    Q = Q / (Q.sum() + 1e-8)
    M, B = Q.shape
    for _ in range(iters):
        Q = Q / (Q.sum(1, keepdim=True) + 1e-8) / M         # component marginal uniform
        Q = Q / (Q.sum(0, keepdim=True) + 1e-8) / B         # sample marginal uniform
    Q *= B
    return Q.t()                                           # [B,M], rows ~sum to 1

def mix_balanced_loss(out, k):
    """Balanced hard-EM: Sinkhorn gives a collapse-free assignment Q; each sample
    pulls up its assigned component(s).  Reliably spreads components across modes."""
    means, logpi = split_mix(out)
    joint = logpi + mix_comp_ll(means, k)                   # [N,M]
    Q = sinkhorn(joint.detach())                           # [N,M] balanced targets
    return -(Q * joint).sum(1).mean()

def mix_pi_reg(out, ctx, is_multi_vec):
    """Per-context mixing-weight entropy: with a FIXED variance the NLL is O(1) per
    sample, so this O(1) regulariser can actually shape pi -- pushed UP on multimodal
    contexts (keep several modes alive) and DOWN on deterministic ones (commit to one)."""
    means, logpi = split_mix(out)
    pi = torch.exp(logpi)
    nctx = is_multi_vec.shape[0]
    sums = torch.zeros(nctx, N_COMP, device=out.device).index_add_(0, ctx, pi)
    cnt = torch.zeros(nctx, device=out.device).index_add_(
        0, ctx, torch.ones_like(ctx, dtype=torch.float))
    present = cnt > 0
    mp = sums[present] / cnt[present, None]
    ent = -(mp * torch.log(mp + 1e-9)).sum(1)
    mmask = is_multi_vec[present]
    loss = out.new_zeros(())
    if mmask.any():
        loss = loss - ent[mmask].mean()
    if (~mmask).any():
        loss = loss + ent[~mmask].mean()
    return loss

def mix_induced(out):
    """Induced next-token distribution: mixture density evaluated at each codebook
    point, normalised over the vocab.  [N,V]."""
    means, logpi = split_mix(out)                           # [N,M,D],[N,M]
    # ||K_v - mu_m||^2 = ||K_v||^2 - 2 K_v.mu_m + ||mu_m||^2
    Kn = (K_raw * K_raw).sum(1)                             # [V]  (=1)
    mn = (means * means).sum(-1)                            # [N,M]
    cross = torch.einsum("vd,nmd->nmv", K_raw, means)       # [N,M,V]
    d2 = Kn[None, None, :] - 2 * cross + mn[:, :, None]     # [N,M,V]
    compll = -0.5 * d2 / VAR[0]
    logmix = torch.logsumexp(logpi[:, :, None] + compll, dim=1)   # [N,V]
    return torch.softmax(logmix, dim=1)

# --------------------------------------------------------------- training ---- #
def train(head, steps=4000, lr=2e-3):
    if head == "mixture":
        steps = 6000
    model = GPT(in_table, V, block_size=1, d_model=128, n_layer=2, n_head=4,
                dropout=0.0, head=head, n_comp=N_COMP).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        x, y = make_batch()
        out = model(x)[:, 0]               # [B, .] at the single position
        if head == "gaussian":
            mu, logvar = split_gauss(out)
            loss = gauss_nll(mu, logvar, K_raw[y])
        elif head == "mixture":
            # deterministic annealing (spread first, then sharpen onto modes) +
            # balanced assignment (no dead components) + per-context pi entropy
            VAR[0] = 2.0 * (0.02 / 2.0) ** (step / steps)
            bal = mix_balanced_loss(out, K_raw[y])
            coef = 1.0 * max(0.3, 1.0 - step / steps)
            loss = bal + coef * mix_pi_reg(out, x[:, 0], is_multi)
        else:                              # kron_decode (Hypothesis A)
            loss = F.cross_entropy(out.reshape(-1, out.size(-1)),
                                   btargets[y].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model

# --------------------------------------------------------------- metrics ---- #
def kl_true_pred(pred):                     # mean KL(true||pred) over contexts
    kls = []
    for ci in range(len(contexts)):
        t = true_dist[ci]
        sup = t > 0
        kl = (t[sup] * (torch.log(t[sup]) - torch.log(pred[ci, sup] + 1e-12))).sum()
        kls.append(kl.item())
    kls = np.array(kls)
    return kls, float(kls[is_multi.cpu().numpy()].mean()), float(kls[~is_multi.cpu().numpy()].mean())

@torch.no_grad()
def eval_gaussian(model, n_samples=2000):
    out = model(ctx_ids[:, None])[:, 0]
    mu, logvar = split_gauss(out)
    induced, _ = gauss_induced(mu, logvar)
    kls, kl_multi, kl_det = kl_true_pred(induced)

    # calibration: mean predicted variance per context
    var = torch.exp(logvar).mean(1).cpu().numpy()
    cal_multi = float(var[is_multi.cpu().numpy()].mean())
    cal_det = float(var[~is_multi.cpu().numpy()].mean())

    # codebook-MAP exact-match on deterministic contexts
    map_tok = induced.argmax(1)
    det_mask = ~is_multi
    det_correct = sum(map_tok[ci].item() == stoi[contexts[ci][1][0]]
                      for ci in range(len(contexts)) if not contexts[ci][2])
    det_acc = det_correct / int(det_mask.sum())

    # mode recovery via sampling + nearest-codebook (||K||=1 -> nearest = max dot)
    sigma = torch.exp(0.5 * logvar)
    recover_tv, valid_rate = [], []
    for ci in range(len(contexts)):
        if not contexts[ci][2]:
            continue
        eps = torch.randn(n_samples, D, device=DEVICE)
        z = mu[ci] + sigma[ci] * eps                        # [S,D]
        snap = (z @ K_raw.T).argmax(1)                      # nearest codebook token
        counts = torch.bincount(snap, minlength=V).float()
        emp = counts / counts.sum()
        tv = 0.5 * (emp - true_dist[ci]).abs().sum().item()
        recover_tv.append(tv)
        sup = set(stoi[w] for w in contexts[ci][1])
        valid_rate.append(sum(int(t.item()) in sup for t in snap) / n_samples)

    # codebook-FREE coherence: does a sample decode to a *valid* word by itself?
    free_mean_valid, free_samp_valid = [], []
    for ci in range(len(contexts)):
        if not contexts[ci][2]:
            continue
        sup = set(contexts[ci][1])
        # mean-inversion (point) free decode
        s_mean = labels_from_vec(mu[ci])
        free_mean_valid.append(1.0 if s_mean in sup else 0.0)
        # sampled free decode
        eps = torch.randn(400, D, device=DEVICE)
        zz = mu[ci] + sigma[ci] * eps
        vs = [labels_from_vec(zz[j]) in sup for j in range(zz.shape[0])]
        free_samp_valid.append(float(np.mean(vs)))

    return {
        "kl_multimodal": round(kl_multi, 4), "kl_deterministic": round(kl_det, 4),
        "pred_var_multimodal": cal_multi, "pred_var_deterministic": cal_det,
        "calibration_ratio": round(cal_multi / cal_det, 2),
        "deterministic_map_acc": round(det_acc, 4),
        "sampling_recover_TV": round(float(np.mean(recover_tv)), 4),
        "sampling_valid_rate": round(float(np.mean(valid_rate)), 4),
        "free_meaninv_valid_rate": round(float(np.mean(free_mean_valid)), 4),
        "free_sampled_valid_rate": round(float(np.mean(free_samp_valid)), 4),
    }, induced

def labels_from_vec(vec):
    """Codebook-FREE decode: reshape a D-vector to 256x32 and read argmax per column
    for the first WLEN positions (fixed-length task)."""
    M = vec.view(D_C, D_P)
    cols = M[:, :WLEN].argmax(0).tolist()
    return bytes(cols).decode("utf-8", errors="replace")

@torch.no_grad()
def eval_hyp_a(model):
    out = model(ctx_ids[:, None])[:, 0]                     # [n_ctx, d_p, C]
    lp = torch.log_softmax(out, dim=-1)
    # induced dist over vocab (product of per-position categoricals)
    score = torch.zeros(len(contexts), V, device=DEVICE)
    for p in range(D_P):
        score += lp[:, p, :][:, btargets[:, p]]
    induced = torch.softmax(score, 1)
    kls, kl_multi, kl_det = kl_true_pred(induced)
    # free per-position argmax decode -> valid word?
    free = out.argmax(-1)                                    # [n_ctx, d_p]
    free_valid = []
    for ci in range(len(contexts)):
        if not contexts[ci][2]:
            continue
        s = labels_to_string(free[ci].tolist())
        free_valid.append(1.0 if s in set(contexts[ci][1]) else 0.0)
    return {"kl_multimodal": round(kl_multi, 4), "kl_deterministic": round(kl_det, 4),
            "free_argmax_valid_rate": round(float(np.mean(free_valid)), 4)}


@torch.no_grad()
def eval_mixture(model, n_samples=2000):
    VAR[0] = DECODE_VAR                                      # decode temperature
    out = model(ctx_ids[:, None])[:, 0]
    means, logpi = split_mix(out)
    pi = torch.exp(logpi)                                    # [N,M]
    induced = mix_induced(out)
    kls, kl_multi, kl_det = kl_true_pred(induced)

    # calibration: entropy of the mixing distribution (nats)
    ent = (-(pi * logpi).sum(1)).cpu().numpy()
    cal_multi = float(ent[is_multi.cpu().numpy()].mean())
    cal_det = float(ent[~is_multi.cpu().numpy()].mean())

    # deterministic MAP exact-match
    det_correct = sum(induced[ci].argmax().item() == stoi[contexts[ci][1][0]]
                      for ci in range(len(contexts)) if not contexts[ci][2])
    det_acc = det_correct / int((~is_multi).sum())

    # OPEN-VOCAB coherent sampling: pick a component ~ pi, decode its mean by
    # per-column argmax -> a coherent word (no codebook used).  Recover modes + freq.
    recover_tv, valid_rate = [], []
    for ci in range(len(contexts)):
        if not contexts[ci][2]:
            continue
        comp = torch.multinomial(pi[ci], n_samples, replacement=True)   # [S]
        emitted = [labels_from_vec(means[ci, m]) for m in range(N_COMP)]
        words = [emitted[int(m)] for m in comp]
        sup = set(contexts[ci][1])
        valid_rate.append(np.mean([w in sup for w in words]))
        # empirical distribution over the true continuations
        emp = np.array([np.mean([w == c for w in words]) for c in contexts[ci][1]])
        tv = 0.5 * float(np.abs(emp - 1.0 / len(contexts[ci][1])).sum()
                         + max(0.0, 1.0 - emp.sum()))     # mass on invalid words
        recover_tv.append(tv)

    return {
        "kl_multimodal": round(kl_multi, 4), "kl_deterministic": round(kl_det, 4),
        "mix_entropy_multimodal": round(cal_multi, 4),
        "mix_entropy_deterministic": round(cal_det, 4),
        "calibration_ratio": round((cal_multi + 1e-9) / (cal_det + 1e-9), 2),
        "deterministic_map_acc": round(det_acc, 4),
        "open_vocab_sampling_TV": round(float(np.mean(recover_tv)), 4),
        "open_vocab_valid_rate": round(float(np.mean(valid_rate)), 4),
    }


def example_modes(model, n=3):
    """For a few multimodal contexts: the true continuations vs the mixture's
    per-component decoded words and weights."""
    out = model(ctx_ids[:, None])[:, 0]
    means, logpi = split_mix(out)
    pi = torch.exp(logpi)
    ex = []
    multi = [ci for ci in range(len(contexts)) if contexts[ci][2]][:n]
    for ci in multi:
        comps = sorted(
            [{"word": labels_from_vec(means[ci, m]), "pi": round(pi[ci, m].item(), 3)}
             for m in range(N_COMP)], key=lambda d: -d["pi"])
        ex.append({"context": contexts[ci][0], "true": contexts[ci][1], "components": comps})
    return ex


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "mix":      # fast iteration on mixture
        m = train("mixture")
        print(json.dumps(eval_mixture(m), indent=2))
        for e in example_modes(m, n=4):
            print(f"  {e['context']}: true={e['true']} -> "
                  f"{[(c['word'], c['pi']) for c in e['components']]}")
        sys.exit(0)

    print("== training Hypothesis A (kron_decode, categorical) ==")
    m_a = train("kron_decode")
    res_a = eval_hyp_a(m_a)
    print("  Hyp A:", res_a)

    print("== training Hypothesis B (gaussian over codec space) ==")
    m_b = train("gaussian")
    res_b, _ = eval_gaussian(m_b)
    print("  Hyp B (single Gaussian):", json.dumps(res_b, indent=2))

    print("== training Hypothesis B+ (MIXTURE of Gaussians over codec space) ==")
    m_c = train("mixture")
    res_c = eval_mixture(m_c)
    ex = example_modes(m_c)
    print("  Hyp B+ (mixture):", json.dumps(res_c, indent=2))
    for e in ex:
        print(f"    {e['context']}: true={e['true']}  ->  "
              f"{[ (c['word'], c['pi']) for c in e['components'] ]}")

    result = {
        "device": DEVICE, "vocab": V, "n_multimodal": N_MULTI,
        "n_deterministic": N_DET, "continuations_per_multi": K, "word_len": WLEN,
        "n_components": N_COMP,
        "hypothesis_a": res_a,
        "hypothesis_b_single": res_b,
        "hypothesis_b_mixture": res_c,
        "examples": ex,
        "headline": {
            "A_kl_multimodal": res_a["kl_multimodal"],
            "single_gauss_kl_multimodal": res_b["kl_multimodal"],
            "mixture_kl_multimodal": res_c["kl_multimodal"],
            "A_open_vocab_valid": res_a["free_argmax_valid_rate"],
            "single_gauss_open_vocab_valid": res_b["free_sampled_valid_rate"],
            "mixture_open_vocab_valid": res_c["open_vocab_valid_rate"],
            "mixture_calibration_ratio": res_c["calibration_ratio"],
        },
    }
    with open("results/hypothesis_b.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nHEADLINE:", json.dumps(result["headline"], indent=2))
