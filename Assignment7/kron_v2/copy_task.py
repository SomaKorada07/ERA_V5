"""
Proof #3: OPEN-VOCABULARY OUTPUT.

The V2 head predicts bytes, not vocabulary ids, so it can emit tokens that were never
in the training vocabulary.  A standard softmax head structurally cannot: any token
outside its |V| output slots has probability identically zero.

Task: a delayed byte-copy.  For a word w, the sequence is  [w, SEP, w]  and the model
must predict w at the position after SEP (route w's identity across the separator and
decode it).  We train on word set A and TEST on a DISJOINT set B of words the model's
weights never saw.  The Kronecker input still encodes B (V1 property); the question is
whether the V2 head can EMIT B (the V2 contribution).

Run: python3 copy_task.py   -> results/copy.json
"""

import json, random, string
import numpy as np
import torch

from codec import build_codec_table, byte_targets, labels_to_string, D_P
from model import GPT

torch.manual_seed(7); np.random.seed(7)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def random_words(n, seed, lo=3, hi=8):
    rng = random.Random(seed)
    out = set()
    while len(out) < n:
        L = rng.randint(lo, hi)
        out.add("".join(rng.choice(string.ascii_lowercase) for _ in range(L)))
    return sorted(out)


TRAIN_W = random_words(1200, seed=1)
TEST_W  = [w for w in random_words(400, seed=999) if w not in set(TRAIN_W)][:300]
SEP = "|"
# One shared id space so the FIXED input codec can encode every word (train+test).
# Training batches only ever use TRAIN_W ids, so the weights never see TEST_W.
vocab = [SEP] + TRAIN_W + TEST_W
stoi = {w: i for i, w in enumerate(vocab)}
codec_table = build_codec_table(vocab)
btargets = torch.tensor(byte_targets(vocab)).to(DEVICE)
SEP_ID = stoi[SEP]
print(f"device={DEVICE}  train_words={len(TRAIN_W)}  test_words={len(TEST_W)}")


def make_batch(words, bs=128):
    ws = [random.choice(words) for _ in range(bs)]
    seq = np.array([[stoi[w], SEP_ID, stoi[w]] for w in ws], dtype=np.int64)
    x = torch.tensor(seq[:, :2]).to(DEVICE)        # [w, SEP]
    y_word = torch.tensor(seq[:, 2]).to(DEVICE)    # target at pos1 = w
    return x, y_word


@torch.no_grad()
def reconstruct_acc(model, words, n=300):
    """Free-byte decode at pos1; exact string match to the target word."""
    model.eval()
    words = words[:n]
    seq = np.array([[stoi[w], SEP_ID] for w in words], dtype=np.int64)
    x = torch.tensor(seq).to(DEVICE)
    out = model(x)                                  # [B,2,d_p,C]
    labels = out[:, 1].argmax(-1).cpu().numpy()     # [B,d_p] predicted bytes at pos1
    recon = [labels_to_string(labels[i]) for i in range(len(words))]
    exact = sum(r == w for r, w in zip(recon, words))
    model.train()
    return exact / len(words), list(zip(words, recon))[:12]


def train(steps=2500, lr=3e-3):
    model = GPT(codec_table, len(vocab), block_size=2, d_model=256,
                n_layer=3, n_head=8, head="kron_decode").to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist = []
    for step in range(steps):
        x, yw = make_batch(TRAIN_W)
        out = model(x)                              # [B,2,d_p,C]
        loss = model.loss_kron(out[:, 1:2], btargets[yw][:, None])  # supervise pos1
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 250 == 0 or step == steps - 1:
            tr, _ = reconstruct_acc(model, TRAIN_W)
            te, ex = reconstruct_acc(model, TEST_W)
            hist.append({"step": step, "train_word_recon": round(tr, 4),
                         "unseen_word_recon": round(te, 4)})
            print(f"  step {step:4d}  loss {loss.item():.4f}  "
                  f"train_recon {tr:.3f}  UNSEEN_recon {te:.3f}")
    return model, hist


if __name__ == "__main__":
    model, hist = train()
    tr, tr_ex = reconstruct_acc(model, TRAIN_W)
    te, te_ex = reconstruct_acc(model, TEST_W)

    # Softmax head: how many TEST words even exist in a vocab built from TRAIN words?
    train_vocab = set(TRAIN_W)
    softmax_representable = sum(w in train_vocab for w in TEST_W) / len(TEST_W)

    result = {
        "device": DEVICE,
        "n_train_words": len(TRAIN_W), "n_test_words": len(TEST_W),
        "history": hist,
        "final_train_word_recon": round(tr, 4),
        "final_unseen_word_recon": round(te, 4),
        "softmax_unseen_representable_frac": softmax_representable,
        "examples_unseen": [{"target": w, "emitted": r} for w, r in te_ex],
        "note": "V2 head reconstructs words never in the training vocabulary. A softmax "
                "head can represent 0% of them (no output slot exists).",
    }
    with open("results/copy.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nUNSEEN-word reconstruction (V2):", round(te, 4))
    print("Softmax head could represent", f"{softmax_representable:.0%}",
          "of unseen words (structural limit).")
    print("examples:", te_ex[:6])
