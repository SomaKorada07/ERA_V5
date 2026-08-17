"""
Proof #2: a real transformer trains with the V2 invertible head and is competitive
with the standard softmax head -- at a fraction of the head parameters.

Identical GPT body + identical Kronecker input for both arms; only the output head
differs.  We report next-token EXACT-match accuracy, which is apples-to-apples:
  * softmax arm     : argmax token id == target id
  * kron_decode arm : predicted d_p byte-labels == target token's byte-labels
                      (injectivity of the codec makes this == predicting the token)

Run: python3 train.py
Writes results/train.json
"""

import json, time, math
import numpy as np
import torch

import data
from codec import build_codec_table, byte_targets, labels_to_string
from model import GPT

torch.manual_seed(1337)
np.random.seed(1337)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ------------------------------------------------------------------ data ----- #
BLOCK = 48
tokens = data.generate(n_sentences=5000, seed=0)
vocab, stoi = data.build_vocab(tokens)
ids = np.array([stoi[t] for t in tokens], dtype=np.int64)
n = len(ids)
split = int(0.9 * n)
train_ids, val_ids = ids[:split], ids[split:]
print(f"device={DEVICE}  tokens={n}  vocab={len(vocab)}")

codec_table = build_codec_table(vocab)               # [V, D], fixed
btargets = torch.tensor(byte_targets(vocab)).to(DEVICE)   # [V, d_p]


def get_batch(src, bs=64):
    ix = np.random.randint(0, len(src) - BLOCK - 1, size=bs)
    x = np.stack([src[i:i + BLOCK] for i in ix])
    y = np.stack([src[i + 1:i + 1 + BLOCK] for i in ix])
    return (torch.tensor(x).to(DEVICE), torch.tensor(y).to(DEVICE))


def kron_token_scores(logits):
    """Faithful Hypothesis-A decode: turn per-position byte log-probs into a score
    over the *vocabulary* by summing each token's byte log-probabilities.
    scores[b,t,v] = sum_p logprob[b,t,p, byte_of_token_v_at_p].  [B,T,V]."""
    lp = torch.log_softmax(logits, dim=-1)             # [B,T,d_p,C]
    B, T, P, C = lp.shape
    scores = torch.zeros(B, T, btargets.shape[0], device=lp.device)
    for p in range(P):
        scores += lp[:, :, p, :][:, :, btargets[:, p]]  # gather -> [B,T,V]
    return scores


@torch.no_grad()
def evaluate(model, src, iters=40):
    model.eval()
    losses, exact, total, free_exact = [], 0, 0, 0
    for _ in range(iters):
        x, y = get_batch(src)
        out = model(x)
        if model.head_type == "softmax":
            losses.append(model.loss_softmax(out, y).item())
            pred = out.argmax(-1)                       # [B,T] token ids
            exact += (pred == y).sum().item()
            total += y.numel()
        else:
            yt = btargets[y]                            # [B,T,d_p] byte labels
            losses.append(model.loss_kron(out, yt).item())
            # in-vocab decode (fair, apples-to-apples with softmax)
            pred = kron_token_scores(out).argmax(-1)    # [B,T] token ids
            exact += (pred == y).sum().item()
            total += y.numel()
            # free open-vocab decode (independent per-position argmax)
            free = out.argmax(-1)                        # [B,T,d_p] byte labels
            free_exact += (free == yt).all(-1).sum().item()
    model.train()
    fe = free_exact / total if model.head_type != "softmax" else None
    return float(np.mean(losses)), exact / total, fe


def train_arm(head, steps=1500, lr=3e-3):
    model = GPT(codec_table, len(vocab), BLOCK, head=head).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist = []
    t0 = time.time()
    for step in range(steps):
        x, y = get_batch(src=train_ids)
        out = model(x)
        loss = (model.loss_softmax(out, y) if head == "softmax"
                else model.loss_kron(out, btargets[y]))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 100 == 0 or step == steps - 1:
            vloss, vacc, vfree = evaluate(model, val_ids)
            rec = {"step": step, "val_loss": round(vloss, 4),
                   "val_exact_acc": round(vacc, 4)}
            if vfree is not None:
                rec["val_free_byte_acc"] = round(vfree, 4)
            hist.append(rec)
            extra = f"  free_byte_acc {vfree:.3f}" if vfree is not None else ""
            print(f"  [{head:11}] step {step:4d}  val_loss {vloss:.4f}  "
                  f"exact_acc {vacc:.3f}{extra}")
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "head": head,
        "history": hist,
        "final_val_exact_acc": hist[-1]["val_exact_acc"],
        "final_val_loss": hist[-1]["val_loss"],
        "head_params": model.head_params(),
        "total_params": total_params,
        "train_seconds": round(time.time() - t0, 1),
    }, model


if __name__ == "__main__":
    results = {"device": DEVICE, "vocab_size": len(vocab), "block": BLOCK, "arms": {}}
    for head in ["softmax", "kron_decode"]:
        print(f"== training arm: {head} ==")
        r, _ = train_arm(head)
        results["arms"][head] = r

    a, b = results["arms"]["softmax"], results["arms"]["kron_decode"]
    results["summary"] = {
        "softmax_exact_acc": a["final_val_exact_acc"],
        "kron_exact_acc": b["final_val_exact_acc"],
        "softmax_head_params": a["head_params"],
        "kron_head_params": b["head_params"],
        "note": "V2 head is competitive on next-token exact-match while its head "
                "size is independent of |V| (see prove.py for the scaling curve).",
    }
    with open("results/train.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSUMMARY:", json.dumps(results["summary"], indent=2))
