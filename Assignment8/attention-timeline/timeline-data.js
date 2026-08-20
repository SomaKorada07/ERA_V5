/* =====================================================================
   ATTENTION TIMELINE — verified data
   Every `date` is the arXiv v1 submission date (or original blog/release),
   read from the primary source. Two dates are forum/blog posts that could
   not be re-verified in-sandbox and are flagged `approx:true`.
   ===================================================================== */

window.__DIMS__ = {
  position: { label: "Position",            color: "var(--dim-position)", blurb: "How token order enters the score" },
  compute:  { label: "Compute · T²",        color: "var(--dim-compute)",  blurb: "The cost of comparing every pair" },
  memory:   { label: "KV-cache memory",     color: "var(--dim-memory)",   blurb: "What decoding must keep per token" },
  context:  { label: "Long-context reach",  color: "var(--dim-context)",  blurb: "Serving far past the trained length" },
  recur:    { label: "Recurrent state",     color: "var(--dim-recur)",    blurb: "Fixed-size memory instead of history" },
  sparse:   { label: "Sparsity · compression", color: "var(--dim-sparse)", blurb: "Read or store fewer positions" }
};

/* Era bands show the field's shifting priority. Inserted before the first
   card whose date is >= the band's `from`. */
window.__ERAS__ = [
  { from:"2016-01", year:"2016–17", text:"<b>Order has to be injected.</b> Remove recurrence and attention becomes permutation-blind — the first thing the Transformer needs is a position signal." },
  { from:"2019-01", year:"2019–20", text:"<b>The T² wall.</b> As context grows the all-pairs score matrix dominates. First wave of answers: don't look at everything (sparse), or don't keep exact history (linear)." },
  { from:"2021-01", year:"2021",    text:"<b>Getting position right, fixing linear memory.</b> Relative position done cleanly (RoPE, ALiBi); the delta rule teaches a fixed state to <i>edit</i>, not just accumulate." },
  { from:"2022-01", year:"2022–23", text:"<b>Serving reality.</b> Now it has to run cheaply for many users: exact-but-IO-aware kernels, KV-cache sharing, and training-free ways to stretch context after the fact." },
  { from:"2024-01", year:"2024",    text:"<b>Recurrence returns at scale.</b> Compress the cache itself (MLA); make the delta rule parallel and gated so linear state finally competes with attention." },
  { from:"2025-01", year:"2025",    text:"<b>Sparsity & compression return — natively trainable.</b> Not an inference bolt-on but built into pre-training; and the boldest position move yet: drop position embeddings entirely." }
];

window.__DATA__ = [
  /* ---------------------------------------------------------------- */
  {
    id:"learned-abs", abbr:"Learned absolute positions", full:"trainable position embeddings",
    date:"Nov 2016", sort:"2016-11", dim:"position",
    source:{ title:"A Convolutional Encoder Model for NMT (Gehring et al.); popularised in ConvS2S, BERT, GPT-1", id:"arXiv:1611.02344", url:"https://arxiv.org/abs/1611.02344" },
    beats:{
      existed:"Attention (and convolutional seq2seq) compares content only. Swap two identical tokens and every projection is identical — the model literally cannot tell position 2 from position 20.",
      problem:"Something outside the dot product has to say <b>where</b> a token is.",
      mechanism:"Learn one vector per absolute slot (0,1,2,…) and add it to the token embedding. Pure lookup: position <i>i</i> gets embedding row <i>i</i>.",
      fixed:"Order is now visible, and the table is fully learned — the model fits whatever positional structure the data actually has.",
      cost:"A lookup table has a hard edge. There is no row for a position you never trained, so the model cannot run beyond its table — the length wall that haunts the rest of this story."
    },
    trade:{
      buys:"Dead simple, fully learned, no hand-designed function.",
      gives:"Zero extrapolation past the trained length; no built-in notion of relative distance; params grow with max length.",
      when:"Fixed, known context you train and serve at the same length — classic BERT/GPT-style encoders and classifiers."
    },
    note:"Often mis-credited to BERT/GPT (2018). The earliest clean origin is Gehring et al., <b>Nov 2016</b> — two years earlier."
  },
  {
    id:"sinusoidal", abbr:"Sinusoidal positions", full:"fixed sine/cosine encoding",
    date:"Jun 2017", sort:"2017-06", dim:"position",
    source:{ title:"Attention Is All You Need — Vaswani et al. (the same paper that defines the scaled dot-product attention above)", id:"arXiv:1706.03762", url:"https://arxiv.org/abs/1706.03762" },
    beats:{
      existed:"A learned table can't extrapolate and its size scales with length. The Transformer wanted a position signal with neither problem.",
      problem:"Encode order with <b>no parameters</b> and a value defined at <i>any</i> position, even lengths never seen.",
      mechanism:"Give each position a fixed vector of sine and cosine waves at geometrically spaced frequencies, and add it to the embedding. Offsets between positions become linear combinations of each other, so relative distance is in principle recoverable.",
      fixed:"Parameter-free, deterministic, and mathematically defined for arbitrarily long sequences.",
      cost:"'Defined at any length' is not 'works at any length'. In practice extrapolation degrades, and because it is an <i>additive absolute</i> scheme, relative distance is only implicit — later beaten by RoPE/ALiBi."
    },
    trade:{
      buys:"No parameters, no length ceiling in the formula, trivial to implement.",
      gives:"Weak real-world extrapolation; relative position only implicit; empirically matched or beaten by learned/relative schemes.",
      when:"A clean parameter-free baseline, or when you simply don't want a position table. Largely superseded in frontier LLMs."
    }
  },
  {
    id:"sparse-tf", abbr:"Sparse Transformers", full:"factorized / strided attention",
    date:"Apr 2019", sort:"2019-04", dim:"sparse",
    source:{ title:"Generating Long Sequences with Sparse Transformers — Child, Gray, Radford, Sutskever (OpenAI)", id:"arXiv:1904.10509", url:"https://arxiv.org/abs/1904.10509" },
    beats:{
      existed:"Standard attention is exact all-to-all — Fact 1 from the top. At tens of thousands of tokens the T² score matrix simply won't fit or finish.",
      problem:"Do we really need every token to look at every other token in every layer?",
      mechanism:"Factorize the dense matrix into a few fixed sparse patterns — 'strided' and 'fixed' — so each position attends to only ~√n others. Different heads/layers use different patterns, so any two positions can still connect in a couple of hops.",
      fixed:"Attention cost drops from O(n²) to O(n√n); sequences of tens of thousands of tokens became trainable on 2019 hardware.",
      cost:"The patterns are hand-designed, not content-adaptive, and no single layer reaches all positions directly — some long-range links only exist indirectly, through depth."
    },
    trade:{
      buys:"First big cut to attention's asymptotic cost; enabled genuinely long inputs (images, audio, long text).",
      gives:"Fixed, non-learned sparsity; indirect long-range routing; a poor fit when the important token is unpredictable.",
      when:"Very long, structurally-local sequences where a fixed pattern is acceptable. Foundational; today's schemes are more flexible."
    },
    fig:"window"
  },
  {
    id:"mqa", abbr:"MQA", full:"Multi-Query Attention",
    date:"Nov 2019", sort:"2019-11", dim:"memory",
    source:{ title:"Fast Transformer Decoding: One Write-Head is All You Need — Noam Shazeer", id:"arXiv:1911.02150", url:"https://arxiv.org/abs/1911.02150" },
    beats:{
      existed:"Sparse attention attacked compute. But generation has a second bill — Fact 2, the KV cache — and it is memory-bandwidth, not FLOPs, that stalls decoding.",
      problem:"Every decode step reloads the whole KV cache from memory; with H separate key/value heads that's a lot of bytes per token.",
      mechanism:"Keep all the query heads, but let them share a <b>single</b> key head and a single value head. The per-token cache shrinks by roughly the head count.",
      fixed:"Dramatically smaller KV cache and far less bandwidth per step — much faster incremental decoding.",
      cost:"One shared K/V is a real capacity cut: some quality loss, and training-from-scratch can be unstable. Too blunt a knob — which is exactly what GQA later softens."
    },
    trade:{
      buys:"Big cache/bandwidth savings; faster decoding.",
      gives:"Reduced K/V capacity → quality hit and instability if pushed all the way to one head.",
      when:"Decode throughput and cache footprint dominate and a small quality hit is fine. Mostly superseded by GQA as the default."
    },
    fig:"heads"
  },
  {
    id:"topk", abbr:"Top-k attention", full:"Explicit Sparse Transformer",
    date:"Dec 2019", sort:"2019-12", dim:"sparse",
    source:{ title:"Explicit Sparse Transformer: Concentrated Attention Through Explicit Selection — Zhao et al.", id:"arXiv:1912.11637", url:"https://arxiv.org/abs/1912.11637" },
    beats:{
      existed:"Sparse Transformers chose <i>where</i> to look with a fixed pattern. But softmax also spreads weight thinly over many irrelevant tokens, diluting focus.",
      problem:"Can each query keep only the keys that actually matter to <i>it</i>, chosen from content rather than a fixed shape?",
      mechanism:"Score all keys, keep the top-k logits per query, set the rest to −∞ before softmax. Selection is data-dependent, so different queries keep different keys.",
      fixed:"Sharper, more interpretable attention — and often better accuracy, because noise from irrelevant tokens is removed.",
      cost:"The catch the lecture stressed: you still <b>score every key</b> before you can pick the best k, so this is a quality method, not an efficiency win. k is a sensitive knob and the hard cutoff is non-smooth."
    },
    trade:{
      buys:"Content-adaptive focus, interpretability, sometimes higher accuracy.",
      gives:"No asymptotic speedup on its own (full scoring still happens); extra hyperparameter; can drop a useful key.",
      when:"You want concentrated, adaptive attention for quality. For real speedups it must be paired with a cheap candidate proposer (window, router, index) — the idea NSA/DSA later build on."
    },
    fig:"topk"
  },
  {
    id:"longformer", abbr:"Sliding-window attention", full:"local window + a few global tokens",
    date:"Apr 2020", sort:"2020-04", dim:"sparse",
    source:{ title:"Longformer: The Long-Document Transformer — Beltagy, Peters, Cohan (formalised it; popularised in production by Mistral 7B, 2023)", id:"arXiv:2004.05150", url:"https://arxiv.org/abs/2004.05150" },
    beats:{
      existed:"Fixed factorized patterns worked but were fiddly. A simpler bet: in language, most of what a token needs is <i>nearby</i>.",
      problem:"Get linear-cost attention with a pattern that's simple, hardware-friendly, and streaming-cache-friendly.",
      mechanism:"Each token attends only to a fixed window of w neighbours (plus, in Longformer, a handful of global tokens). Stack L such layers and the effective receptive field grows to ~L×w.",
      fixed:"Attention cost falls to O(n·w); pairs beautifully with KV caching for long-document and streaming use. Mistral 7B (2023) shipped it at w=4096 in a frontier model.",
      cost:"No direct long-range link inside one layer — distant dependencies must propagate through depth and can attenuate. The far past outside the window is simply unreachable."
    },
    trade:{
      buys:"Linear, simple, streaming-friendly, strong when locality dominates.",
      gives:"Long-range reach only via depth; genuinely distant facts can be lost.",
      when:"Long inputs where locality carries most of the signal, or efficient streaming/chat. A default building block — often combined with global tokens or sinks."
    },
    note:"Mistral's own blog is widely cited as 27 Sep 2023 but was unreachable here; the primary-source-verified date is the Mistral 7B arXiv paper, 10 Oct 2023.",
    fig:"window"
  },
  {
    id:"linear", abbr:"Linear attention", full:"Transformers are RNNs",
    date:"Jun 2020", sort:"2020-06", dim:"recur",
    source:{ title:"Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention — Katharopoulos et al. (Performer/FAVOR+ followed, Sep 2020)", id:"arXiv:2006.16236", url:"https://arxiv.org/abs/2006.16236" },
    beats:{
      existed:"Sparsity kept softmax but read fewer keys. What if the expensive part <i>is</i> softmax? Recall the lecture's arithmetic: softmax ties every score to a shared denominator, so you must keep the individual old keys.",
      problem:"Remove softmax and the query factors out — you can pre-combine the past into one running state before the query even arrives.",
      mechanism:"Replace softmax(QKᵀ)V with a kernel map φ: attention becomes φ(Q)(φ(K)ᵀV). Compute φ(K)ᵀV first → O(n). For causal decoding it's literally a recurrence over a fixed-size state matrix S that doesn't grow with tokens.",
      fixed:"Linear time and O(1)-per-step generation with constant memory — after 10 tokens or a million, S is the same shape. Both bills, in one move.",
      cost:"A fixed feature map is a coarse stand-in for softmax: weaker recall on retrieval-style tasks. And a pure add-only state can only accumulate — it cannot correct or erase an old association. That specific failure is the next card."
    },
    trade:{
      buys:"O(n) compute, constant-memory decoding, no growing KV cache.",
      gives:"Lossy compressed memory; weaker exact recall; interference as associations pile up.",
      when:"Very long sequences or throughput-bound generation where a fixed-size state is acceptable and exact recall isn't the priority."
    },
    fig:"state"
  },
  {
    id:"deltanet", abbr:"The delta rule (DeltaNet)", full:"linear transformers as fast-weight programmers",
    date:"Feb 2021", sort:"2021-02", dim:"recur",
    source:{ title:"Linear Transformers Are Secretly Fast Weight Programmers — Schlag, Irie, Schmidhuber", id:"arXiv:2102.11174", url:"https://arxiv.org/abs/2102.11174" },
    beats:{
      existed:"Linear attention gave us a compact fixed state — but an add-only state carries the old answer forward. Ask key A for 40, then write 55, and a read returns 40+55 = 95, not 55.",
      problem:"A useful memory must be able to <b>revise</b>, not just pile on.",
      mechanism:"Treat the state as a fast-weight memory and apply the delta (error-correcting) rule: read what the key currently returns, take the difference from the target, and write only that correction, scaled by a learned β. 40 → measure the +15 gap → write 15 → 55.",
      fixed:"Far better use of a finite state: reliable associative recall and updateable memory at the same fixed size.",
      cost:"The delta update is inherently sequential — each step depends on the current state — so the original form was hard to parallelize across a sequence, a training-speed bottleneck that stalled it at scale for years (fixed in 2024)."
    },
    trade:{
      buys:"Linear-attention efficiency plus real, editable associative memory.",
      gives:"Sequential update → hard to train fast; still a compressed state, not exact history.",
      when:"You want linear efficiency but need dependable in-context retrieval rather than lossy accumulation."
    },
    fig:"delta"
  },
  {
    id:"rope", abbr:"RoPE", full:"Rotary Position Embedding",
    date:"Apr 2021", sort:"2021-04", dim:"position",
    source:{ title:"RoFormer: Enhanced Transformer with Rotary Position Embedding — Su et al.", id:"arXiv:2104.09864", url:"https://arxiv.org/abs/2104.09864" },
    beats:{
      existed:"Absolute schemes — learned or sinusoidal — bolt position on <i>additively</i> and mostly encode 'I am token 8'. What attention actually cares about is 'how far apart are we?'.",
      problem:"Put <b>relative</b> distance directly inside the query–key dot product, without a position table.",
      mechanism:"Rotate each query and key by an angle proportional to its position (in 2D dimension-pairs, at several rates). In the dot product the absolute rotations cancel and only the difference (i−j) survives — so the score depends on relative distance. Move both tokens 10 places: same gap, same angle, same score.",
      fixed:"A strong relative-position inductive bias with attention that naturally decays with distance, no added parameters, and compatibility with linear attention. It became the default (LLaMA, Qwen, Mistral, GPT-NeoX…).",
      cost:"The rotation is <i>defined</i> at any position but the model only learned the patterns inside its training length. Push far beyond it and the rotations combine in ways it never saw — the long-context problem NTK-aware scaling and YaRN exist to patch."
    },
    trade:{
      buys:"Relative structure, distance decay, parameter-free, linear-attention compatible — today's standard.",
      gives:"Native extrapolation past training length is limited; small per-dim compute step.",
      when:"The default for modern decoder LLMs — and the base that every context-extension trick below assumes."
    },
    note:"Frequently mis-dated to 2022 or 2023 — those are revisions. The verified v1 is <b>April 2021</b>.",
    fig:"rope"
  },
  {
    id:"alibi", abbr:"ALiBi", full:"Attention with Linear Biases",
    date:"Aug 2021", sort:"2021-08", dim:"position",
    source:{ title:"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation — Press, Smith, Lewis", id:"arXiv:2108.12409", url:"https://arxiv.org/abs/2108.12409" },
    beats:{
      existed:"Even RoPE extrapolates poorly far past its training length. What if position weren't an embedding at all?",
      problem:"Train on short sequences and still work on much longer ones at inference.",
      mechanism:"No position embeddings. Add a fixed linear penalty proportional to query–key distance straight onto the scores, with a per-head slope. This bakes in a recency bias directly.",
      fixed:"Genuinely strong length extrapolation (train 1024, test 2048+ at matching perplexity), plus ~11% less time and memory than sinusoidal. Dead simple.",
      cost:"The recency bias is hard-coded, so it can hurt tasks that need strong attention to the distant past; less flexible than RoPE, which is why RoPE (+ scaling) won the frontier and ALiBi stayed niche (BLOOM, MPT)."
    },
    trade:{
      buys:"Excellent train-short/test-long extrapolation; faster and lighter; trivial.",
      gives:"Baked-in recency bias; weaker for long-range-dependency tasks; less dominant now.",
      when:"When extrapolation is the priority and a recency bias fits the task."
    }
  },
  {
    id:"flash", abbr:"FlashAttention", full:"IO-aware exact attention",
    date:"May 2022", sort:"2022-05", dim:"compute",
    source:{ title:"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness — Dao, Fu, Ermon, Rudra, Ré", id:"arXiv:2205.14135", url:"https://arxiv.org/abs/2205.14135" },
    beats:{
      existed:"Every approximation so far changed the <i>math</i> to dodge T². But standard attention was also just badly implemented: it writes the whole N×N matrix to slow GPU memory.",
      problem:"The bottleneck is memory traffic (HBM reads/writes), not arithmetic — can we compute the <b>exact</b> same softmax attention while barely touching HBM?",
      mechanism:"Tile Q/K/V and fuse the whole operation into one kernel that keeps blocks in fast on-chip SRAM, using online softmax to stitch tiles together without ever materialising the full matrix. Recompute in the backward pass instead of storing.",
      fixed:"Large wall-clock speedups and memory linear in sequence length — with <b>bit-for-bit exact</b> results. Longer contexts became practical without changing the model at all.",
      cost:"No accuracy cost; the price is engineering (hardware-specific kernels) and slightly more FLOPs from recomputation. Crucially it shrinks the attention <i>computation</i>, not the KV <i>cache</i> — so it composes with, rather than replaces, MQA/GQA/MLA."
    },
    trade:{
      buys:"Exact attention, much faster, linear memory — essentially free quality.",
      gives:"Complex kernels; doesn't touch the KV-cache bill; extra recompute FLOPs.",
      when:"Almost always, for training and prefill on modern GPUs. It's a kernel, not an architecture — the honest odd one out on this timeline."
    }
  },
  {
    id:"gqa", abbr:"GQA", full:"Grouped-Query Attention",
    date:"May 2023", sort:"2023-05", dim:"memory",
    source:{ title:"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints — Ainslie et al. (Google)", id:"arXiv:2305.13245", url:"https://arxiv.org/abs/2305.13245" },
    beats:{
      existed:"MQA (one K/V head) saved the most cache but hurt quality and could be unstable; full MHA is expensive to serve. The knob was all-or-nothing.",
      problem:"Get most of MQA's savings with almost none of the quality loss — and without retraining from scratch.",
      mechanism:"Split the query heads into G groups; each group shares one K/V head (MHA = G heads, MQA = 1). Then 'uptrain' an existing MHA checkpoint into GQA with a small fraction of the original compute.",
      fixed:"Near-MHA quality at most of MQA's cache/bandwidth savings, a tunable dial, and cheap conversion of pretrained models. It became the modern serving default (Llama-2 70B onward).",
      cost:"It reduces how much you store <i>per token</i> but still stores something for <b>every</b> token — the cache still grows linearly with context. GQA lowers the slope; it doesn't stop the line."
    },
    trade:{
      buys:"Near-MHA quality, big cache savings, tunable, cheap to retrofit.",
      gives:"Larger cache than MQA; linear growth with context remains.",
      when:"The strong practical baseline for serving decoder LLMs. Necessary, not sufficient, for genuinely long context."
    },
    fig:"heads"
  },
  {
    id:"ntk", abbr:"NTK-aware scaling", full:"NTK-aware scaled RoPE",
    date:"Jun 2023", sort:"2023-06", dim:"context", approx:true,
    source:{ title:"'NTK-Aware Scaled RoPE…' — bloc97, r/LocalLLaMA (a forum post, later formalised by YaRN)", id:"reddit r/LocalLLaMA · thread 14lz7j5", url:"https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/" },
    beats:{
      existed:"RoPE is defined past its training length but behaves badly there. The first fix, linear Position Interpolation, squeezes all frequencies equally and blurs fine local detail.",
      problem:"Extend a RoPE model's context <b>with no fine-tuning</b>, without wrecking perplexity or losing high-frequency detail.",
      mechanism:"Don't scale all rotary frequencies the same. Change the RoPE base ('theta') so interpolation is spread unevenly: high-frequency dimensions are scaled little, low-frequency dimensions more — an NTK-flavoured intuition about which frequencies a network learns.",
      fixed:"Training-free context extension (e.g. LLaMA to 8k+) with far less perplexity loss than linear interpolation, from a one-constant change.",
      cost:"A heuristic — still below fine-tuned quality, and the static form needs the target length known in advance (spawning 'Dynamic NTK'). A patch, not a principled solution — which is the opening for YaRN."
    },
    trade:{
      buys:"Zero-training context extension; one-line change.",
      gives:"Heuristic quality; below fine-tuned methods; static version needs a fixed target length.",
      when:"You need to stretch an existing RoPE model right now and can't fine-tune."
    },
    note:"Originated as a <b>reddit post</b>, not a paper. Commonly cited as late June 2023, but the exact post timestamp could not be re-verified in this environment — treat the day as approximate."
  },
  {
    id:"yarn", abbr:"YaRN", full:"Yet another RoPE extensioN",
    date:"Aug 2023", sort:"2023-08", dim:"context",
    source:{ title:"YaRN: Efficient Context Window Extension of Large Language Models — Peng, Quesnelle, Fan, Shippole", id:"arXiv:2309.00071", url:"https://arxiv.org/abs/2309.00071" },
    beats:{
      existed:"NTK-aware was a training-free heuristic; earlier fine-tuned methods (PI) needed lots of tokens and steps. Neither was both cheap and best-in-class.",
      problem:"Extend RoPE context to high quality with the least possible fine-tuning.",
      mechanism:"Combine 'NTK-by-parts' — interpolate some frequency bands, extrapolate others — with an attention <b>temperature</b> adjustment that rescales logits to keep behaviour stable at long range. Then fine-tune only briefly.",
      fixed:"State-of-the-art context extension with ~10× fewer tokens and ~2.5× fewer steps than prior methods, and it can even reach beyond the fine-tuning length. Widely adopted (Qwen long-context and many open models).",
      cost:"Unlike pure NTK it still needs some fine-tuning; adds a temperature hyperparameter; and it's RoPE-specific — it inherits RoPE's assumptions rather than rethinking position."
    },
    trade:{
      buys:"Best quality-per-fine-tuning-token among RoPE-scaling methods; extends beyond its own tune length.",
      gives:"Needs a short fine-tune; extra hyperparameter; RoPE-only.",
      when:"The go-to for extending a RoPE model's context when you can afford a brief fine-tune."
    }
  },
  {
    id:"sinks", abbr:"Attention sinks", full:"StreamingLLM",
    date:"Sep 2023", sort:"2023-09", dim:"context",
    source:{ title:"Efficient Streaming Language Models with Attention Sinks — Xiao, Tian, Chen, Han, Lewis", id:"arXiv:2309.17453", url:"https://arxiv.org/abs/2309.17453" },
    beats:{
      existed:"To run forever you'd like to just drop old KV entries with a rolling window. But naively evicting the <i>earliest</i> tokens makes quality collapse.",
      problem:"Why does dropping the first few tokens break the model — and can we stream indefinitely without retraining?",
      mechanism:"Diagnosis: models dump a large chunk of attention mass onto the first few tokens — 'attention sinks' — as a no-op place to park probability. Fix: permanently keep those few sink tokens' KV, plus a rolling recent window, and evict the middle.",
      fixed:"Constant memory and stable perplexity over millions of tokens — unbounded streaming with no fine-tuning.",
      cost:"It is <b>not</b> long-context recall: the evicted middle is gone forever, so it can't retrieve arbitrary past facts. The truly attended context is still window-sized — stability, not memory."
    },
    trade:{
      buys:"Bounded memory and coherence over endless streams, training-free.",
      gives:"No recall of the dropped past; effective context stays window-sized.",
      when:"Long-running chat/streaming where you need stability and fixed memory, not exact recall of the far past."
    }
  },
  {
    id:"mla", abbr:"MLA", full:"Multi-head Latent Attention",
    date:"May 2024", sort:"2024-05", dim:"memory",
    source:{ title:"DeepSeek-V2 — MLA is introduced inside the model report (not a standalone paper)", id:"arXiv:2405.04434", url:"https://arxiv.org/abs/2405.04434" },
    beats:{
      existed:"GQA slowed cache growth by storing fewer heads, but it still keeps a full-width K/V per token, and the line still climbs.",
      problem:"Cut the cache far below GQA/MQA <b>without</b> the quality cost of throwing heads away.",
      mechanism:"Down-project keys and values into a shared low-rank <b>latent</b> vector and cache only that compressed latent per token; reconstruct K/V on the fly via up-projection (which can be folded into neighbouring weights). A decoupled RoPE path keeps position working alongside the compression.",
      fixed:"KV cache far smaller than MQA/GQA while matching or beating MHA quality — a genuine improvement on both axes, and strong long-context serving economics.",
      cost:"Real architectural complexity (latent projections, RoPE decoupling, weight absorption) and it must be designed in from pre-training — you can't cheaply bolt it onto an existing MHA checkpoint the way GQA allows."
    },
    trade:{
      buys:"Much smaller cache <i>and</i> MHA-level quality; excellent long-context serving.",
      gives:"Complexity; a pre-training commitment, not a retrofit.",
      when:"You control pre-training and are optimising aggressively for long-context, high-throughput inference where cache size is the binding limit."
    },
    fig:"compress"
  },
  {
    id:"cope", abbr:"CoPE", full:"Contextual Position Encoding",
    date:"May 2024", sort:"2024-05b", dim:"position",
    source:{ title:"Contextual Position Encoding: Learning to Count What's Important — Golovneva, Wang, Weston, Sukhbaatar (Meta/FAIR)", id:"arXiv:2405.18719", url:"https://arxiv.org/abs/2405.18719" },
    beats:{
      existed:"Every position scheme so far counts <i>raw tokens</i>. So a model literally cannot address 'the i-th sentence' or 'the 3rd item' — position is blind to content.",
      problem:"Let position count meaningful units — words, sentences, items — not just token slots.",
      mechanism:"Make position <b>context-dependent</b>: compute a gate on each key and increment the position counter only on tokens the model selects. Positions become fractional and are interpolated over learned embeddings, so 'distance' can mean 'two sentences back'.",
      fixed:"Solves selective-copy, counting and state-tracking tasks where standard PEs fail outright, and improves language- and code-modelling perplexity.",
      cost:"Heavier than RoPE/ALiBi (extra gating computation) and far less battle-tested at scale — a research-stage idea, not yet a production default."
    },
    trade:{
      buys:"Abstraction-level, content-aware addressing; unlocks counting/structure tasks.",
      gives:"More compute and complexity; unproven at frontier scale.",
      when:"Tasks where token-count position is the bottleneck — counting, state-tracking, structured/algorithmic work."
    },
    note:"A strong idea not on the required list, included because it names a limitation <i>every</i> other position scheme shares — position that can't see content."
  },
  {
    id:"pdelta", abbr:"Parallel DeltaNet", full:"delta rule, parallel over sequence length",
    date:"Jun 2024", sort:"2024-06", dim:"recur",
    source:{ title:"Parallelizing Linear Transformers with the Delta Rule over Sequence Length — Yang, Wang, Zhang, Shen, Kim", id:"arXiv:2406.06484", url:"https://arxiv.org/abs/2406.06484" },
    beats:{
      existed:"The delta rule (2021) fixed linear attention's memory but its update was sequential — so it couldn't train fast, and sat on the shelf while attention scaled.",
      problem:"Make the delta update parallel across the sequence so it can train on modern hardware at LLM scale.",
      mechanism:"A chunkwise-parallel reformulation (a WY-style representation) that computes the same delta-rule recurrence in parallel blocks instead of token-by-token.",
      fixed:"DeltaNet finally trains efficiently and scales (demonstrated to 1.3B) — the step that turned a 2021 idea into a practical LLM backbone.",
      cost:"Still a fixed-size recurrent state, so extreme-precision long-range retrieval can trail full attention. It also had no adaptive way to <i>forget</i> stale context — the gap Gated DeltaNet closes next."
    },
    trade:{
      buys:"Editable linear-time memory that actually trains fast at scale.",
      gives:"Fixed state (not exact history); no global forgetting yet.",
      when:"You want a linear-time recurrent backbone with reliable associative recall and can adopt custom chunkwise kernels."
    }
  },
  {
    id:"gated-delta", abbr:"Gated DeltaNet", full:"delta rule + adaptive gating",
    date:"Dec 2024", sort:"2024-12", dim:"recur",
    source:{ title:"Gated Delta Networks: Improving Mamba2 with Delta Rule — Yang, Kautz, Hatamizadeh (NVIDIA)", id:"arXiv:2412.06464", url:"https://arxiv.org/abs/2412.06464" },
    beats:{
      existed:"Two complementary tools existed apart: Mamba-2's gating erases the <i>whole</i> state quickly (good for forgetting), and the delta rule makes <i>surgical</i> key-value edits (good for precision). Neither alone is ideal.",
      problem:"Combine fast, global forgetting with precise, targeted updates in one linear recurrence.",
      mechanism:"One update with two terms: a data-dependent gate α that adaptively decays the whole state, plus the delta rule β for targeted associative writes — trained with a hardware-efficient chunkwise-parallel algorithm.",
      fixed:"Beats both Mamba-2 and plain DeltaNet on language modelling, reasoning, in-context retrieval and length extrapolation; hybrids with a few sliding-window/attention layers go further.",
      cost:"More mechanism than either parent, and it's still a fixed-size state — so pinpoint long-range retrieval can still trail full attention, which is why real systems interleave it with occasional exact-attention layers."
    },
    trade:{
      buys:"Fast forgetting <i>and</i> precise memory edits, linear-time, competitive with attention on many tasks.",
      gives:"Extra complexity; fixed state; not a full replacement for exact attention.",
      when:"A linear-time backbone needing both stale-context decay and precise updates — a stronger Mamba-2 drop-in, usually in a hybrid stack."
    },
    note:"'Recurrence returning' in one paper: the 2021 delta rule, made parallel in mid-2024, now fused with Mamba-2 gating."
  },
  {
    id:"nsa", abbr:"NSA", full:"Native Sparse Attention (DeepSeek)",
    date:"Feb 2025", sort:"2025-02", dim:"sparse",
    source:{ title:"Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention — Yuan et al. (DeepSeek-AI)", id:"arXiv:2502.11089", url:"https://arxiv.org/abs/2502.11089" },
    beats:{
      existed:"Most sparse attention was bolted onto a pre-trained dense model at inference, and often wasn't even fast on GPUs. Top-k still paid to score everything (that catch again).",
      problem:"Make sparsity <b>trainable end-to-end</b> and genuinely hardware-fast — in training <i>and</i> inference.",
      mechanism:"Three branches merged by learned gates: coarse <b>token compression</b> (block summaries), fine-grained <b>token selection</b> (top blocks), and a local <b>sliding window</b> — with blockwise, hardware-aligned kernels so the sparsity turns into real speedups. It compresses history and selects only the top blocks to read, cheaply.",
      fixed:"Sparse attention trainable from scratch, with large prefill/decode speedups on long context while matching or beating full attention quality — this is 'DeepSeek's compressed sparse attention'.",
      cost:"Substantial implementation complexity (custom kernels, multiple branches, gating) and it's a pre-training architectural commitment, not a drop-in inference patch. Compression can lose token-level detail; approximate selection can miss a block."
    },
    trade:{
      buys:"End-to-end trainable sparsity with real hardware speedups and full-attention-level quality on long context.",
      gives:"High complexity; must be designed into pre-training.",
      when:"You are pre-training a long-context model and want native, hardware-aligned sparsity rather than an inference-time approximation."
    },
    fig:"compress"
  },
  {
    id:"dsa", abbr:"DSA", full:"DeepSeek Sparse Attention (V3.2-Exp)",
    date:"Sep 2025", sort:"2025-09", dim:"sparse",
    source:{ title:"DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention — DeepSeek-AI (tech report + repo)", id:"github deepseek-ai/DeepSeek-V3.2-Exp", url:"https://github.com/deepseek-ai/DeepSeek-V3.2-Exp" },
    beats:{
      existed:"NSA proved native sparsity but is a from-scratch architecture. Most already-trained frontier models are dense — retraining them from zero for sparsity is not an option.",
      problem:"Add fine-grained sparse attention to an <i>existing</i> strong dense model via cheap continued training, at production scale.",
      mechanism:"Two lightweight parts on top of MLA: a <b>lightning indexer</b> (a few small heads, ReLU, FP8-friendly) that cheaply ranks which past tokens matter, then <b>top-k selection</b> that runs exact attention only over those tokens. It's the top-k idea made efficient by a cheap proposer — exactly the fix top-k always needed.",
      fixed:"Fine-grained sparse attention at production scale with near-unchanged output quality, added to V3.1 by continued training rather than a full retrain — big long-context efficiency gains.",
      cost:"Still approximate — the indexer can mis-rank and drop a useful token — and it leans on MLA's shared-latent design (it cites NSA for the requirement that KV entries be shared across query heads); it's a targeted efficiency layer, not a rethink of the model."
    },
    trade:{
      buys:"Cheap-to-add, production-proven sparse attention on an existing model with minimal quality loss.",
      gives:"Approximate selection; depends on the MLA/indexer machinery.",
      when:"You have a strong dense long-context model and want efficiency via continued training, not a from-scratch sparse architecture."
    },
    note:"Distinct from NSA (Feb 2025): NSA is trained from scratch; DSA is a lighter indexer-plus-top-k added by continued training. Date anchored to the repo's initial commit (29 Sep 2025) — the tech report carries no printed date."
  },
  {
    id:"drope", abbr:"DroPE", full:"dropping positional embeddings",
    date:"Dec 2025", sort:"2025-12", dim:"context",
    source:{ title:"Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings — Gelberg, Eguchi, Akiba, Cetin (Sakana AI-affiliated)", id:"arXiv:2512.12167", url:"https://arxiv.org/abs/2512.12167" },
    beats:{
      existed:"NTK-aware and YaRN stretch RoPE, but every one of them still <i>relies</i> on an explicit position scheme — and that reliance may be exactly what caps length generalization.",
      problem:"Extend usable context past pre-training length without costly long-context fine-tuning — by questioning whether explicit position embeddings should be there at all.",
      mechanism:"Argument: PEs help a model converge during training but hurt generalization at test time. So <b>remove</b> them after pre-training (with a brief recalibration), and lean on the causal mask alone to imply order (the 'NoPE' intuition). Claims near-seamless zero-shot context extension.",
      fixed:"Reported zero-shot context extension with no long-context fine-tuning, preserving in-context ability — reportedly beating specialised long-context architectures and RoPE-scaling methods.",
      cost:"Very new and largely unproven in the wild; still needs the post-training recalibration; and dropping explicit position is a strong bet that trades a known tool for a claim."
    },
    trade:{
      buys:"Zero-shot context extension without long-context fine-tuning; keeps in-context ability.",
      gives:"Bleeding-edge and unvalidated at scale; requires recalibration; a bold architectural bet.",
      when:"Experimental context extension of an existing model when you want to skip long-context fine-tuning and can run cutting-edge methods."
    },
    note:"'DroPE' is ambiguous. This is the LLM context-extension paper (Dec 2025) — the one matching the course's description of a positional recalibration done before annealing to reach far longer context. Do not confuse it with <b>DRoPE</b> (Directional RoPE, Mar 2025), an unrelated autonomous-driving method. The course also refers to DroPE as an internal step in its own 'V4' model whose exact algorithm it states is not publicly established — so treat any precise mechanism as a hypothesis."
  }
];
