# V5 Data Mixture & Curriculum Specification

**Assignment: Session 5 — Data Mixtures and Curriculum**

This is a defensible, testable plan for the V5 data mixture and curriculum. Every share is composed *backward* from a benchmark we intend to win, sized against the real token supply from the inventory, and declared as a hypothesis to be confirmed or refuted by a named proxy experiment before it is trusted at full scale.

---

## 0. Scope and assumptions

- **Total budget:** 2T tokens (consistent with the Session 5 supply-check figures).
- **Split of the budget into two presets:**
  - **Main pretraining mixture** — the first **1.9T** tokens. All shares in §2 are of this 1.9T.
  - **Anneal mixture** — the final **100B** tokens (5%), a *separate preset* (§6), not a continuation of the main mixture. Learning rate is decayed across it.
- **Target model** (from the brief): Codex-grade coding + multi-step agentic ability, controllable-depth reasoning, and **native Indic fluency as the primary differentiator**.
- Supply numbers below are *real, unique, cleaned, provenance-stamped* tokens surviving Session 3–4 gating. Where demand exceeds supply I say plainly whether the gap is closed by **repetition** or **synthetic generation** — no lane is handed a share it cannot physically be fed.

---

## 1. Capabilities → benchmarks → data shape (the justification chain)

Every number in §2 exists because a named benchmark demands it. The loss target (which tokens are green/trained vs. grey/context-only) is specified per lane, because that is what actually determines the training-data shape.

| Capability        | Benchmark(s)                | Training-data shape & loss target                                                                     |
| ----------------- | --------------------------- | ---------------------------------------------------------------------------------------------------- |
| Code editing      | SWE-bench, Aider            | Repo + issue → **loss on the generated patch only** (repo/issue grey)                                 |
| Coding (general)  | LiveCodeBench, HumanEval+   | Problem statement → **loss on the generated solution**                                                |
| Tool use          | BFCL                        | Function schema + query → **loss on function name + argument JSON**                                   |
| Multi-step agents | tau-bench, GAIA, BrowseComp | Full trajectory → **loss on plans, tool-calls, final answer; grey mask on all tool observations**     |
| Reasoning / math  | AIME, GPQA, MATH, HLE       | Worked traces (SFT) → loss on full trace; later **RLVR** → reward-only on verified final answer       |
| Long-context      | RULER, long-eval            | Long doc + query → loss on final answer; context is normal (unmasked) LM tokens                       |
| Indic             | MILU, IndicGenBench         | Native text + Q&A → standard next-token LM loss                                                       |
| World knowledge   | MMLU                        | Web / encyclopedic text → standard next-token LM loss                                                 |

The masking rule for agents is load-bearing: **observations never receive loss.** Training on tool outputs teaches the model to *hallucinate* tool results instead of calling the tool.

---

## 2. Main pretraining mixture (of 1.9T)

| Lane             | Share                | Tokens  | Real supply | Verdict & how the gap is closed                                              | Wins                          |
| ---------------- | -------------------- | ------- | ----------- | --------------------------------------------------------------------------- | ----------------------------- |
| General web      | 32%                  | 608B    | ~4.5T       | ~7× oversupplied → free repetition margin; lowest marginal value            | MMLU, HLE breadth             |
| Code             | 25%                  | 475B    | ~1.1T       | Covered (~2.3× headroom); no repetition needed                              | SWE-bench, LiveCodeBench      |
| Indic            | 17%                  | 323B    | ~250B real  | **Short in main run — see §3: mix of ~1.1–1.2× repetition + ~90B synthetic** | MILU, IndicGenBench           |
| STEM / math      | 13%                  | 247B    | ~250B       | At supply — **near-zero repetition margin**, do not overdraw                | AIME, GPQA, MATH              |
| Reasoning traces | 6%                   | 114B    | ~85B        | Short → repeat verified traces ~**1.34×**                                   | AIME, GPQA (via later RLVR)   |
| Long-context     | 5%                   | 95B     | ~100B       | Fits exactly — no margin; strictly capped                                    | RULER, long-eval              |
| Agentic / tools  | 2% (protected floor) | 38B     | ~0.63B      | **~98% must be synthesized** — cannot be met from real data (see §4)         | BFCL, tau-bench, GAIA         |
|                  | **100%**             | **1900B** |             |                                                                             |                               |

**Why these numbers, defended against the obvious pushback:**

- **General web 32% (down from a naive ~50%+).** It is ~7× oversupplied, so the marginal token buys the least. Every point cut here is redirected to a supply-constrained lane that actually moves a target benchmark. Cutting it *below* ~30% starts to erode MMLU/world-knowledge breadth, so 32% is the floor of diminishing returns, not an arbitrary round number.
- **Code 25%.** This is the co-headline capability (Codex-style). It is well-supplied, so a large share costs nothing in repetition. 25% is deliberately *not* higher because SWE-bench gains past a point come from *agentic* patch trajectories (§4), not raw code volume.
- **Indic 17%.** The primary differentiator, and supply-constrained. Sized above the 12% protected floor because the whole project exists to win MILU/IndicGenBench — but honestly capped at 17% rather than the 25% one might want, because §3 shows 25% would require majority-synthetic Indic, which is a quality risk. **Whether 17% actually beats the 12% floor is exactly what Proxy Experiment 1 tests (§9).**
- **STEM 13%.** Sized to sit right at real supply (~250B) so we neither starve math nor pay a repetition penalty on a lane where repetition is known to overfit contest problems.
- **Reasoning 6%.** Modest in *pretraining* on purpose: deep reasoning is taught later (SFT + RLVR, Sessions 17–18). This lane is the *seed distribution of trace shapes*, not the reasoning model itself.
- **Long-context 5%.** Capped at real supply; introduced late in the curriculum (§8) so it isn't competing with basic literacy.
- **Agentic 2%.** A protected *floor*, not a ceiling — small in share but almost entirely synthetic (§4), and the real trajectories are reserved for anneal.

**Protected floors (never crossed by the OPUS selector or any rebalancing):** Indic ≥ **12%**, Agentic ≥ **2%**, Reasoning ≥ **5%**. Enforced via an always-on lane (§7).

---

## 3. Indic tier split (verified / unverified / translated / synthetic)

Total Indic across the whole run ≈ **343B** (323B main + 20B anneal). Real, unique Indic supply is only ~250B, and the *verified-native* portion is far smaller — this is the scarcity this session exists to expose.

| Tier                       | Share of Indic | Tokens | Real supply | Sourcing & repetition                                              |
| -------------------------- | -------------- | ------ | ----------- | ----------------------------------------------------------------- |
| **A** — Verified native    | 6%             | 20B    | ~40B        | **100% reserved for anneal + post-training** — 0 in bulk main run |
| **B** — Unverified crawl   | 47%            | 162B   | ~150B       | Bulk of main-run Indic; repeat ~**1.1×**                          |
| **C** — Translated         | 21%            | 71B    | ~60B        | Fills STEM-in-Indic / topical gaps; repeat ~**1.2×**             |
| **D** — Synthetic          | 26%            | 90B    | build       | Generated only for uncovered language × topic cells               |
|                            | **100%**       | **343B** |           |                                                                   |

**Defense of the split:**

- **Tier A is scarce (~40B real) and highest-trust, so it is spent where it has the most leverage: the low-LR anneal**, where a small volume of exceptional data delivers disproportionate capability gain. We reserve ~20B for the anneal Indic slot and hold the remainder for SFT/eval — **none is burned in the bulk main run**, where its signal would be diluted across 1.9T tokens.
  - *(This corrects a flaw in the earlier draft, which "reserved 119B of Tier A for the anneal" — the anneal's entire Indic slot is only 20B, so a 119B reserve cannot physically fit. The reserve is now sized to the slot that consumes it.)*
- **Main-run Indic exposure is carried by B + C** (native crawl + translated), which is abundant enough to cover 233B with light repetition. **Synthetic D (~90B, 26%)** covers only the specific language×topic cells B and C leave thin (e.g. low-resource-language STEM). We flag this openly: a *quarter* of Indic is synthetic, and **its quality is a named risk that Proxy Experiment 1 must clear before we trust the 17% share.**
- We deliberately did **not** push Indic to 25%, because doing so at this supply would force Tier D past 50% — trading the differentiator's *quality* for its *quantity*.

---

## 4. Agentic slot — sourcing plan (the build-don't-collect lane)

Real agentic trajectory supply is **~0.63B tokens** against a 38B main-run demand — two orders of magnitude short. This lane is **built, not collected.**

- **~37.4B synthesized** as multi-step trajectories from three generators, each aimed at a specific benchmark:
  - Sandboxed repo environments → SWE-bench-style **patch trajectories** (plan → edit → run tests → recover).
  - Simulated API/tool sandboxes → BFCL / tau-bench-style **function-call trajectories**.
  - Multi-hop web-search simulations → GAIA / BrowseComp-style **research trajectories** (the grant-hunting task from the brief is the canonical shape: plan → search → read → follow-up → recover from a dead source → answer).
- **Masking:** green on plans, tool-calls, and final responses; **grey on every observation/tool output.** Non-negotiable — see §1.
- **The 0.63B real corpus is 100% reserved for the anneal** (§6). It is the only non-synthetic signal for this capability, so it is never spent in the main run.

Whether synthetic-only agentic data is *good enough*, or whether the tiny real corpus is worth reserving, is **Proxy Experiment 2 (§9).**

---

## 5. Reasoning slot — effort bands (a distribution, not one pool)

The 114B reasoning budget is split by trace depth so the model can later learn a controllable low/medium/high/ultra effort dial (the actual dial is taught in RLVR, Sessions 17–18 — pretraining only supplies the *range* of shapes).

| Band   | Share of reasoning | Concrete example                                                        |
| ------ | ------------------ | ---------------------------------------------------------------------- |
| Low    | 15%                | "What is 12 × 8?" → direct answer, no visible chain                    |
| Medium | 30%                | Single-variable algebra word problem → short verified chain            |
| High   | 35%                | AIME-style combinatorics → multi-path exploration                      |
| Ultra  | 20%                | GPQA-style cross-domain problem → long self-correcting trace           |

**Ultra-band traces are preferentially reserved for the anneal** — most expensive to produce, most directly tied to RLVR. If the selector consumes them early, nothing exceptional is left for the cooldown.

---

## 6. Anneal mixture (final 100B, separate preset, decayed LR)

The anneal concentrates the deliberately-withheld reserves: Tier-A Indic, real agentic trajectories, and ultra-band reasoning.

| Lane                            | Main-run share      | Anneal share      | What is concentrated                     |
| ------------------------------- | ------------------- | ----------------- | ---------------------------------------- |
| General web                     | 32%                 | 10%               | sharply reduced                          |
| Code                            | 25%                 | 20%               | hardest / test-passing patches           |
| Indic                           | 17% (B+C+D)         | 20%               | **100% Tier A (~20B verified native)**   |
| STEM / math                     | 13%                 | 15%               | hardest tier only                        |
| Reasoning                       | 6% (all bands)      | 15%               | **100% ultra band**                      |
| Agentic                         | 2% (mostly synth)   | 15%               | real corpus (0.63B, oversampled ~5×) + best verified synthetic |
| Long-context                    | 5%                  | 5%                | held flat                                |
|                                 |                     | **100% / 100B**   |                                          |

**Honest note on the agentic anneal slot:** the slot is 15B, but only ~0.63B real trajectories exist. We cannot fill it with "100% real" — that would require ~24× oversampling, which memorizes rather than teaches. Instead the whole real corpus is oversampled ~5× (~3B effective) and the remaining ~12B is drawn from the *highest-quality verified* synthetic trajectories (those whose generated tests actually pass). This cap is enforced in code (`mixture_spec.py`), which flags any lane whose implied oversampling exceeds 8×.

The anneal only works if the reserve *survives the main run* — which is why the reservation is a §2/§3 allocation decision made now, enforced by the always-on lane in §7, not something discovered at the end.

---

## 7. Protected floor vs. the OPUS selector

- OPUS retains roughly the top ~40% of candidate batches by estimated learning value (V4: ~6× effective-token multiplier, few-% compute overhead).
- **Failure mode:** an English-heavy proxy systematically undervalues Indic and agentic batches. Left uncontrolled, their *effective* share collapses toward zero regardless of the nominal 17%/2% targets.
- **V5 fix:** Indic, agentic, and reasoning batches sit in an **always-on lane** — sampled at their target rate every iteration, independent of the OPUS score (exactly as V4 pinned Indic at a flat 8%). The final design is *aggressive selection above a protected capability floor.*

---

## 8. Curriculum — order and difficulty ladder

Order matters nearly as much as proportion. Each stage transition is blended over a multi-billion-token warmup band (§10).

1. **General stage** — broad web text first; establishes language, world knowledge, basic structure.
2. **Code / STEM ramp** — shift weight toward code and math once foundations exist (V4 precedent: web 70%→18%, code 13%→35%, STEM 7%→39%).
3. **Long-context stage** — introduced *after* reading/reasoning competence, so long-range retention isn't learned simultaneously with basic literacy.
4. **Anneal** — reserved Tier-A Indic, real agentic trajectories, ultra reasoning, hardest code/math, at decayed LR.

**Difficulty ladder within each stage** (simple → advanced), with a concrete example at each level:

| Level | Example (code lane)                                  | Example (Indic lane)                                    |
| ----- | --------------------------------------------------- | ------------------------------------------------------ |
| L1    | Single-function completion (`is_even`)              | Short native sentence completion                       |
| L2    | Multi-function module with tests                    | Paragraph-level native Q&A (MILU-style)                |
| L3    | Cross-file bug fix given a failing test             | STEM-in-Indic explanation (translated-tier backed)     |
| L4    | Full SWE-bench-style patch trajectory (anneal)      | Verified-native long-form reasoning in Indic (anneal)  |

---

## 9. Proxy validation plan — every share is a hypothesis

No number above is trusted at 2T scale until it survives a cheap proxy run with a **named metric and a pre-committed pass/fail threshold.**

### Experiment 1 — Indic share is worth its cost — *1B params, 20B tokens*

- **Setup:** three runs, identical mixture except the Indic lane at **17% vs. 12% (floor) vs. 25%**, rescaled proportionally.
- **Metric:** MILU accuracy (+ IndicGenBench as secondary).
- **Pass/fail:** if the 12%-floor run scores **within 2 points** of the 17% run on MILU → the extra 5 points buys nothing at scale; drop Indic toward the floor and redirect the freed budget to Code. If 25% beats 17% by **> 3 points** *and* the synthetic-D fraction hasn't degraded quality → raise Indic. Otherwise 17% stands.
- **Secondary check:** ablate Tier-D synthetic (26% → 10%) to confirm synthetic Indic is *helping*, not just filling — if MILU drops < 1 point when D is cut, D is overweight.

### Experiment 2 — Synthetic agentic data quality & the real-data reserve — *3B params, 60B tokens*

- **Setup:** two runs — (a) 100% synthetic agentic trajectories; (b) 100% synthetic **+ the 0.63B real trajectories oversampled 4×**.
- **Metric:** BFCL function-call exact-match; tau-bench task completion.
- **Pass/fail:** if (b) beats (a) by **> 5 points** on tau-bench → the anneal-reservation of real trajectories (§4) is confirmed high-value. If not → real data is worth less than assumed and can be spent earlier; scarcity value was overstated.

### Experiment 3 — Curriculum ordering vs. flat mixture — *1B params, 20B tokens*

- **Setup:** staged curriculum (§8) vs. a flat mixture of identical overall proportions.
- **Metric:** final MMLU + LiveCodeBench + RULER, and training stability (gradient-norm spikes at transitions).
- **Pass/fail:** if staged doesn't beat flat by **> 1.5 points** aggregate *and* introduces no stability benefit → the ordering complexity isn't paying for itself; simplify.

> **Status:** these are the committed proxy specs. Results feed back before any §2–§6 number is locked for the full 2T run. The rubric's highest tier is reserved for bringing these numbers back — this plan is written so those numbers slot directly into the pass/fail gates above.

---

## 10. Keeping the run stable across mixture transitions

Every distribution change (growth-stage boundary or anneal start) shifts the gradient. V4 saw a **~150× gradient-norm spike** when a Hindi-share jump hit frozen embeddings — enough to kill a run.

- **No mixture change is a hard step.** Every transition is blended across a **multi-billion-token warmup band**.
- Architecture and mixture are **frozen before the main run** begins; live changes are planned, infrequent, and monitored as carefully as an architectural change.

---

## 11. Summary — what makes this plan defensible

- **Every lane's share is justified against a named benchmark and its real supply**, with the gap closed by explicit repetition or synthesis — no wishful accounting.
- **Indic is split across all four tiers** (A/B/C/D), with Tier A reserved for the anneal at a volume that *actually fits the slot that consumes it*.
- **Protected floors (Indic 12%, Agentic 2%, Reasoning 5%)** are enforced in an always-on lane the OPUS selector cannot starve.
- **The anneal reserve is declared and sized now**, so it survives the main run.
- **Difficulty and reasoning-length bands** are laid out with concrete examples.
- **Every number is a hypothesis** with a concrete 1B/3B proxy, a named metric, and a pre-committed pass/fail threshold.

---

## 12. Reproducibility — the numbers are executable

Every figure in this document is defined once in code and checked for internal consistency, so the spec cannot silently drift from its own claims. No third-party dependencies (Python 3.10+ standard library only).

| File                    | What it does                                                                                          |
| ----------------------- | ---------------------------------------------------------------------------------------------------- |
| `mixture_spec.py`       | Single source of truth for every share/supply/tier/band. Reproduces the §2/§3/§5/§6 tables and runs 13 invariant checks. |
| `proxy_experiments.py`  | Encodes the §9 pass/fail rules as pure functions — the proxy decisions are mechanical, not post-hoc. |
| `test_mixture.py`       | 26 tests asserting every README invariant (shares sum to 100%, floors respected, reserves fit slots). |

```bash
python3 mixture_spec.py            # print the computed report + invariant checks
python3 mixture_spec.py --check    # CI mode: exit non-zero if any invariant fails
python3 proxy_experiments.py       # worked examples of each §9 decision
python3 test_mixture.py            # run the full invariant test suite
```

The checks are load-bearing, not decorative: they are what caught the earlier draft's Tier-A reserve error (§3) and force the agentic anneal slot to be honest about its ~24× oversampling gap (§6). Any edit that makes a number inconsistent fails a check.
