# How Attention Works Now — a chronological, honestly-traded timeline

An interactive single-page explainer of the attention mechanisms covered in **ERA V5 · Session 8 — Modern Attention Variants**.

It starts from **scaled dot-product attention** and then presents every later idea **in the order it was actually launched**, each as a response to a cost the previous mechanism created:

```
what existed  →  the problem  →  the new mechanism  →  what it fixed  →  the new trade-off
```

Read top to bottom and the field's priorities visibly move: exact global attention → cheaper decode memory → better position → longer context → recurrence returning → sparsity & compression returning.

- **Live site:** https://REPLACE_WITH_PAGES_URL
- **Repository:** https://github.com/REPLACE_WITH_REPO
- Static, dependency-free (`index.html` + `timeline-data.js` + `render.js`). Works from `file://` or any static host.

---

## Why chronological, and why "honestly traded"

Standard attention was not "bad" and then replaced by something "better". It solved a real problem (content-based, exact, all-to-all mixing) and created two new bills — **T² compute** and a **per-conversation KV cache** — plus one structural gap: attention is order-blind, so **position** must be injected separately. Almost every mechanism on the timeline attacks one of those and pays somewhere else. Each card therefore answers three questions honestly: **what it buys, what it gives up, and when you'd actually choose it.** If a card looked like a free lunch, it went back for re-checking.

---

## The chronology and its sources

Every date is the **arXiv v1 submission date** (read from the paper's submission history), or the original blog/release where there is no paper. Dates are exactly where a confident-sounding agent goes wrong, so each was checked against the primary source rather than a secondary summary. Two entries are forum/blog posts that could **not** be re-verified from the primary source in the build environment and are flagged **≈ approximate**.

| # | Technique | Date (v1) | Primary source | Link |
|---|-----------|-----------|----------------|------|
| — | *Scaled dot-product attention (the anchor)* | Jun 2017 | Vaswani et al., *Attention Is All You Need* | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| 01 | Learned absolute positions | **Nov 2016** | Gehring et al., *A Convolutional Encoder Model for NMT* (earliest clean origin; also ConvS2S, BERT, GPT-1) | [arXiv:1611.02344](https://arxiv.org/abs/1611.02344) |
| 02 | Sinusoidal positions | **Jun 2017** | Vaswani et al., *Attention Is All You Need* | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| 03 | Sparse Transformers (factorized/strided) | **Apr 2019** | Child, Gray, Radford, Sutskever, *Generating Long Sequences with Sparse Transformers* | [arXiv:1904.10509](https://arxiv.org/abs/1904.10509) |
| 04 | MQA — Multi-Query Attention | **Nov 2019** | Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* | [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) |
| 05 | Top-k attention (Explicit Sparse Transformer) | **Dec 2019** | Zhao et al., *Explicit Sparse Transformer* | [arXiv:1912.11637](https://arxiv.org/abs/1912.11637) |
| 06 | Sliding-window attention | **Apr 2020** | Beltagy, Peters, Cohan, *Longformer* (formalised it; popularised in production by Mistral 7B, 2023) | [arXiv:2004.05150](https://arxiv.org/abs/2004.05150) |
| 07 | Linear attention | **Jun 2020** | Katharopoulos et al., *Transformers are RNNs* (Performer/FAVOR+ followed, Sep 2020, [arXiv:2009.14794](https://arxiv.org/abs/2009.14794)) | [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) |
| 08 | The delta rule (DeltaNet) | **Feb 2021** | Schlag, Irie, Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers* | [arXiv:2102.11174](https://arxiv.org/abs/2102.11174) |
| 09 | RoPE — Rotary Position Embedding | **Apr 2021** | Su et al., *RoFormer* | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| 10 | ALiBi — Attention with Linear Biases | **Aug 2021** | Press, Smith, Lewis, *Train Short, Test Long* | [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) |
| 11 | FlashAttention (exact, IO-aware) | **May 2022** | Dao, Fu, Ermon, Rudra, Ré, *FlashAttention* | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |
| 12 | GQA — Grouped-Query Attention | **May 2023** | Ainslie et al., *GQA* | [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) |
| 13 | NTK-aware scaled RoPE **≈** | **~Jun 2023** | bloc97, r/LocalLLaMA post (forum; later formalised by YaRN) — *date approximate, see note* | [reddit thread 14lz7j5](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/) |
| 14 | YaRN | **Aug 2023** | Peng, Quesnelle, Fan, Shippole, *YaRN* | [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) |
| 15 | Attention sinks (StreamingLLM) | **Sep 2023** | Xiao, Tian, Chen, Han, Lewis, *Efficient Streaming Language Models with Attention Sinks* | [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) |
| 16 | MLA — Multi-head Latent Attention | **May 2024** | DeepSeek-AI, *DeepSeek-V2* (MLA introduced inside the model report) | [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |
| 17 | CoPE — Contextual Position Encoding | **May 2024** | Golovneva, Wang, Weston, Sukhbaatar, *Contextual Position Encoding* | [arXiv:2405.18719](https://arxiv.org/abs/2405.18719) |
| 18 | Parallel DeltaNet | **Jun 2024** | Yang, Wang, Zhang, Shen, Kim, *Parallelizing Linear Transformers with the Delta Rule* | [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) |
| 19 | Gated DeltaNet | **Dec 2024** | Yang, Kautz, Hatamizadeh (NVIDIA), *Gated Delta Networks: Improving Mamba2 with Delta Rule* | [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) |
| 20 | NSA — Native Sparse Attention ("DeepSeek's compressed sparse attention") | **Feb 2025** | Yuan et al. (DeepSeek-AI), *Native Sparse Attention* | [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) |
| 21 | DSA — DeepSeek Sparse Attention (V3.2-Exp) | **Sep 2025** | DeepSeek-AI, *DeepSeek-V3.2-Exp* (tech report + repo) | [github: DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) |
| 22 | DroPE — dropping positional embeddings | **Dec 2025** | Gelberg, Eguchi, Akiba, Cetin (Sakana AI-affiliated), *Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings* | [arXiv:2512.12167](https://arxiv.org/abs/2512.12167) |

### Related mechanisms referenced inline (dates also verified)
- **Performer / FAVOR+** — Sep 2020, [arXiv:2009.14794](https://arxiv.org/abs/2009.14794)
- **Mamba** — Dec 2023, [arXiv:2312.00752](https://arxiv.org/abs/2312.00752); **Mamba-2** (in *Transformers are SSMs*) — May 2024, [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
- **RWKV** — May 2023, [arXiv:2305.13048](https://arxiv.org/abs/2305.13048); **RetNet** — Jul 2023, [arXiv:2307.08621](https://arxiv.org/abs/2307.08621)
- **Mistral 7B** (sliding window in production) — paper Oct 2023, [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)

---

## Dates flagged as approximate, and disambiguations

- **NTK-aware scaled RoPE (#13):** originated as a **reddit post** by *bloc97* on r/LocalLLaMA, not a paper. It is commonly cited as **late June 2023**, but the exact post timestamp could **not** be re-verified in the build environment (reddit and the Wayback Machine were unreachable). Marked **≈** on the site. The YaRN paper credits it as *bloc97, 2023*.
- **Mistral 7B sliding window:** Mistral's own blog is widely cited as **27 Sep 2023** but was unreachable (HTTP 403). The primary-source-verified date is the **arXiv paper, 10 Oct 2023**; the site anchors the sliding-window card on **Longformer (Apr 2020)**, which formalised the mechanism, and notes Mistral's production adoption.
- **DeepSeek-V3.2-Exp / DSA (#21):** the tech report PDF carries no printed date; the **29 Sep 2025** date is anchored to the GitHub repository's initial commit — a strong but indirect primary-source signal.
- **"DroPE" is ambiguous (#22):** the site uses the **LLM context-extension** paper *Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings* ([arXiv:2512.12167](https://arxiv.org/abs/2512.12167), Dec 2025). Do **not** confuse it with **DRoPE — Directional Rotary Position Embedding** ([arXiv:2503.15029](https://arxiv.org/abs/2503.15029), Mar 2025), an unrelated autonomous-driving method.

## Corrections caught during fact-checking

Per the assignment's warning ("check every launch date… catch mistakes, including mine"):

1. **RoPE is April 2021**, not 2022 or 2023 — those are later arXiv revisions of the same paper. This is a very commonly repeated error.
2. **Learned absolute position embeddings predate BERT/GPT (2018).** The earliest clean origin is **Gehring et al. (Nov 2016)**, ~2 years earlier.
3. **YaRN authors** are Peng, Quesnelle, Fan, Shippole (no "Sharkey").
4. **"DeepSeek's compressed sparse attention"** is not one thing: **NSA** (Feb 2025, trained from scratch) and **DSA** (Sep 2025, added to an existing dense model by continued training) are distinct techniques; both are shown.

### On the course's own model (LightningLM "V4"/"V5")

Session 8 also discusses techniques from an internal architecture — a **Memory Stream**, a **`DDDGDDDG`** depth schedule, and "**V4 Compressed Sparse Attention**". These are not dated public launches, so they are **not** pins on the timeline; the closest *published* anchors are shown instead (MLA, NSA/DSA, Gated DeltaNet). The session notes themselves state that the exact **DroPE** algorithm "is not established" in their record — so the DroPE card uses the matching public paper and flags any precise mechanism as a hypothesis, in keeping with the notes.

---

## Method

- Dates were verified by fetching arXiv **submission-history** pages (the authoritative record of the `[v1]` timestamp) and primary repositories, not secondary blog summaries.
- The colour that marks each card's **cost dimension** (position / compute / KV memory / long-context / recurrent state / sparsity-compression) uses a categorical palette validated for colour-blind separation and contrast; colour is always paired with a written label, so it is never the only cue.

## Run locally

```bash
cd attention-timeline
python3 -m http.server 8000
# open http://localhost:8000
```

## Files

- `index.html` — page shell, design tokens (light/dark), and the interactive standard-attention demo
- `timeline-data.js` — the verified, sourced data for every mechanism
- `render.js` — timeline renderer, legend filters, and inline-SVG mini-diagrams

---

*Built for ERA V5, Session 8. Corrections welcome — the timeline is only as good as its citations.*
