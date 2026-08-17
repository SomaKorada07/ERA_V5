# Kronecker Embeddings V2 — The Invertible Codec & the Vocabulary-Free Head

**Problem solved (from the V2 brief): #5 —**
> *"Kronecker is forward deterministic (same word → same embedding). How do I make a
> reverse of this? If we can do this, then we can get rid of the final head as well!
> Then we can have a vocab of 1M as well without any issues!"*

This is exactly the paper's own flagged open question (V1 §8.5, "Hypothesis A:
tied-head Kronecker decoding"). V2 turns that hypothesis into a proven, trained result.

---

## The one-sentence idea

The V1 codec is **already exactly invertible** — reshape `κ(b)` into a `256 × 32`
matrix and every column is a scaled one-hot of the byte at that position, so `argmax`
down each column returns the bytes. Therefore the output side never needs a
`d_model → |V|` head: the model predicts the **next token's codec** (32 parallel
256-way byte classifiers) and we decode it back to bytes. **No vocabulary head ⇒ the
head size is constant in `|V|`, and the model can emit tokens it never saw in training.**

```
FORWARD (V1):   "run"  --encode-->  κ ∈ ℝ^{256×32}   (column p = one-hot of byte p)
INVERSE (V2):   κ      --argmax per column-->  bytes  --utf8-->  "run"     [EXACT]
OUTPUT HEAD:    h ∈ ℝ^{d_model}  -->  32 × 257 logits  -->  argmax per slot  -->  next token
```

---

## Three proofs (all reproduced by the code, real numbers below)

### 1. The codec is *exactly* invertible — a theorem, not a hope (`prove.py`)
Each position `p` maps to exactly one column of the `256 × 32` matrix, so `(byte,pos)`
pairs never collide and `decode(encode(s)) == s` for every string `≤ 32` bytes.
- **70/70** vocabulary tokens round-trip with **0 failures** (incl. `café`, `北京`, `emoji😀`, `π`).
- **52/52** tokens decode exactly from the V2 head's target labels.

> Note: this is *injectivity of the encoding*. It is different from the V1 cosine
> collisions (`compute`/`commute`) which are about two *different* tokens being *close*,
> not about `κ` losing information. `κ` is genuinely one-to-one.

### 2. The head is O(1) in vocabulary size (`prove.py`)
Head parameters at `d_model = 768`:

| Vocab | Softmax head | V2 Kron head | V2 is |
|------:|-------------:|-------------:|:-----:|
| 1,000 | 768 K | 6.3 M | 0.1× (softmax wins at tiny V) |
| 50,000 | 38.4 M | 6.3 M | **6.1× smaller** |
| 131,072 | 100.7 M | 6.3 M | **15.9× smaller** |
| 262,144 | 201.3 M | 6.3 M | **31.8× smaller** |
| **1,000,000** | **768.0 M** | **6.3 M** | **121× smaller** |

The softmax head grows linearly and hits ~768 M params at a 1M vocab; the V2 head is
flat. "A vocab of 1M without any issues" — literally.

### 3. It trains, and it works (`train.py`, `copy_task.py`)

**A real GPT (identical body + identical Kronecker input; only the head differs)**
on a templated-grammar corpus — next-token *exact-match* accuracy:

| Head | Val exact-acc | Head params |
|------|:-------------:|:-----------:|
| Softmax (baseline) | **0.113** | 9,984 |
| V2 Kron-Decode | **0.103** | — |

Parity (both bounded by the grammar's irreducible entropy) — **the invertible head is
not the bottleneck**. Decoding uses the paper's Hypothesis-A rule: score each vocab
token by summing its per-position byte log-probabilities.

**Open-vocabulary output** (`copy_task.py`) — a delayed byte-copy `[w, SEP, w]` trained
on 1,200 words and tested on **300 disjoint random words the weights never saw**:

| | V2 Kron-Decode | Softmax |
|---|:---:|:---:|
| Reconstruct **trained** words | **100%** | (in-vocab only) |
| Reconstruct **unseen** words | **71%** | **0% (structurally impossible)** |

The 71% are *exact full-word* matches of random letter strings (e.g. `agnudp→agnudp`,
`aogcmh→aogcmh`); the misses are 1–2 bytes off (`ameouby→ameokbl`), the expected effect
of the lossy `D→d_model` projection plus per-position independence. A softmax head can
represent **0** of these because no output slot exists for them.

---

## Proof 4 — Hypothesis B: fixing the multimodal case (`hypothesis_b.py`)

Proof 3a used a *faithful* decode (score each vocab token by its byte log-probs) to reach
parity. But that leans on a vocabulary. The residual weakness of the invertible head is
**codebook-free / ambiguous** decoding: because it predicts byte positions **independently**,
a point-decode of an ambiguous next token blends the options into garbage (this is why
`free_byte_acc` was 0). The V1 paper's §8.5 "Hypothesis B" proposes the fix — predict a
**distribution over codec space**, not a point. We implement it and test it on a *controlled*
task with known ground truth: 12 multimodal contexts (each followed uniformly by 3 fixed
5-letter words) + 6 deterministic contexts.

Three heads, same body:

| Head | in-vocab KL to truth ↓ | open-vocab valid-word rate ↑ | ambiguity signalled? |
|------|:---:|:---:|:---:|
| **A — categorical** (`kron_decode`) | **0.03** (fine *in-vocab*) | 0.33 (blends) | no |
| **B — single Gaussian** (paper-literal) | 16.5 | 0.001 | no (ratio 1.0) |
| **B⁺ — mixture of Gaussians** (the fix) | **0.12** | **0.98** | **yes (ratio ~16,000)** |

What the numbers say:

- **A single diagonal Gaussian mean-collapses.** A unimodal Gaussian cannot represent
  several modes, so it predicts the conditional *mean* — a blend. KL blows up (16.5) and its
  open-vocab decode is essentially never a valid word. This is a real (negative) result with
  a clear cause, not a bug.
- **A mixture of Gaussians over codec space fixes it.** With deterministic annealing (spread
  the components, then sharpen), balanced (Sinkhorn) assignment (no dead components), and a
  per-context mixing-weight entropy term, the mixture places **one component on each mode**.
  On multimodal contexts it recovers all three continuations with weights ≈ 1/3 each:

  ```
  ctx00  true = [tixlz, wxuqa, oyhub]
         mix  =  wxuqa(0.33)  oyhub(0.32)  tixlz(0.32)  + a 0.03 leftover
  ```
- **It signals ambiguity.** The mixing-weight entropy is ~1.2 nats on multimodal contexts and
  ~0 on deterministic ones (calibration ratio ~16,000× vs the single Gaussian's 1.0). The head
  says *how many* futures are plausible.
- **It samples into coherent, open-vocab words.** Pick a component ∝ π and decode its mean:
  98% of samples are valid real words (vs the categorical head's independent-position decode,
  which blends). The recovered distribution matches ground truth to total-variation 0.03.

So the honest, full picture: the categorical head already represents multimodal targets fine
*when a vocabulary is available*; the mixture head is what makes **codebook-free, coherent,
calibrated** multimodal generation work. The single Gaussian — the literal paper proposal — is
the instructive baseline that shows why a *mixture* is needed. The mixture head is still O(1)
in |V|.

### Proof 4b — the mixture head generating real text (`lm_hypb.py`)

Folding the mixture head back into the **main LM task** (the same templated-grammar corpus as
Proof 3a), we generate autoregressively by sampling a component ∝ π and decoding its mean in
codec space. Sampled continuations are grammatical:

```
the quick fox    -> builds each scholar while the gentle machine follows the lantern
a silent river   -> watches a signal the golden sparrow carries a lantern each
the ancient kingdom -> a ancient lantern follows each fox while another golden engine
```

The components line up with **part of speech** — the head learned the grammar's branching:

| after… | top sampled components (word · π) |
|--------|-----------------------------------|
| `the` | gentle·.18, lantern·.13, golden·.12, quick·.09, silent·.09 (adjectives) |
| `the quick` | lantern·.13, fox·.09, machine·.09, scholar·.09, harbor·.09 (nouns) |
| `the ancient river` | questions·.10, carries·.10, crosses·.09, measures·.09 (verbs) |

**Codebook-free token validity (no vocabulary used at decode): mixture 82.5% vs Hypothesis A's
independent per-position sampling 2.5%.** Hyp A codebook-free output is byte soup
(`wu ev lagtnre e qeelon…`); the mixture emits coherent words because a single component is a
single coherent point in codec space. (Codebook-*snap* readout — nearest real codec row — gives
the clean sentences above; codebook-*free* is the same but decoded straight from the 256×32 grid,
with occasional 1-byte slips like `lanternr`.)

The mixture head is still O(1) in |V|.

---

## What this buys you (the V2 payoff)

1. **Drop the `lm_head`.** Output side becomes a fixed codec + one small projection —
   symmetric with the V1 input side.
2. **Unbounded vocabulary at inference.** Emit any UTF-8 string ≤ 32 bytes, including
   words absent from training. Vocab of 1M (or ∞) costs the same as vocab of 50k.
3. **Deployment.** With V1's input savings *and* V2's output savings, a deployed model
   carries transformer body + two small projections + the byte mapping. No embedding
   matrix, no unembedding matrix.
4. **Calibrated, open-vocab multimodal generation** via the mixture head (Proof 4): the
   model expresses how many next tokens are plausible and samples coherent words for each.

## Honest limitations

- **Per-position independence (categorical head).** The `kron_decode` head factorizes the
  next token as independent byte positions, so on *ambiguous* targets a free byte-argmax
  blends modes (`free_byte_acc = 0` on the LM task, while the faithful vocab-restricted
  decode reaches parity). **Proof 4 fixes this** with the mixture-Gaussian (Hypothesis B⁺)
  head, which samples coherent, calibrated, open-vocab multimodal outputs.
- **Mixture training is finicky at tiny scale.** Getting all K modes per context needs
  deterministic annealing + balanced assignment + a mixing-weight entropy term; a plain
  mixture NLL mode-collapses. The single diagonal Gaussian (paper-literal) cannot represent
  modes at all and is included as the instructive baseline.
- **Lossy projection** (`D=8192 → d_model`) caps exact byte recall on long/rare words;
  the 71% unseen-word number rises with `d_model`.
- **Break-even.** The V2 heads only win on parameters above ~8k vocab (see table). It is
  a large-vocab / frontier-scale technique, exactly the regime V1 targets.

---

## Run it

```bash
python3 -m pip install torch numpy      # torch 2.13, MPS/CPU both fine
python3 prove.py         # proof 1 & 2: exact invertibility + param scaling  -> results/prove.json
python3 train.py         # proof 3a: LM comparison softmax vs V2 head          -> results/train.json
python3 copy_task.py     # proof 3b: open-vocabulary output on unseen words    -> results/copy.json
python3 hypothesis_b.py  # proof 4:  multimodal fix (categorical/Gaussian/mixture) -> results/hypothesis_b.json
python3 lm_hypb.py       # proof 4b: mixture head generating real text            -> results/lm_hypb.json
```

## Files

| File | What |
|------|------|
| `codec.py` | The invertible codec: `encode_matrix` / `decode_matrix` (the proof object), model-facing tables, V2 head targets/decoder |
| `model.py` | Small GPT; Kronecker input; selectable `softmax` / `kron_decode` / `gaussian` / `mixture` heads |
| `data.py` | Templated-grammar corpus (learnable next-token structure) |
| `prove.py` | Exact-invertibility round-trip + head-label consistency + param-scaling |
| `train.py` | LM comparison of the two heads (same body, same input) |
| `copy_task.py` | Open-vocabulary output: reconstruct unseen words |
| `hypothesis_b.py` | Hypothesis B: single Gaussian (baseline) vs mixture-of-Gaussians (fix) on a controlled multimodal task |
| `lm_hypb.py` | Mixture head folded into the LM task: autoregressive generation + per-context component (part-of-speech) analysis |
| `results/*.json` | Raw outputs from the runs above |
| `index.html` | Visual write-up (interactive codec/decoder + charts) |

Built on Kronecker Embeddings V1 (Rohan Shravan, The School of AI, 2026).
