# V5 Training Data Execution System (Session 6)

A small but **complete, reproducible, auditable** training-data execution system: it
turns cleaned documents into the exact stream a training loop consumes — token
windows, loss masks, attention/segment ids, position ids, mixture tags, packed
microbatches — and then **proves** the stream is correct, replayable, and
reconstructable.

The point is not scale. The point is that every token the model sees can be
explained, replayed, and audited, and that the run can crash and recover without
skipping or repeating a single batch.

- **Zero dependencies** — Python 3.10+ standard library only.
- **One command** — `python3 run_demo.py` regenerates everything.
- **Nothing hardcoded** — the evidence bundle is produced by the implementation.

---

## Quick start

```bash
cd Assignment6
python3 run_demo.py          # runs the full demonstration -> submission_artifacts/
python3 tests/test_invariants.py   # 19 invariant tests (or: python3 -m pytest tests/)
```

`run_demo.py` exits `0` only if all 20 evidence checks pass.

---

## What it produces

```
submission_artifacts/
  run.log                 # complete event log with [PASS]/[FAIL] markers
  evidence.json           # machine-readable pass/fail + artifact pointers
  evidence.md             # human-readable evidence table
  performance.json        # throughput, packing utilization, mixture compliance
  manifests/              # one immutable manifest per shard + mixture_timeline.json
  ledgers/                # consumption.jsonl, learning.jsonl, opus.jsonl,
                          #   firewall.json, fork.json, audit.json
  checkpoints/            # model+data state (main, recover, fork branches)
```

---

## The path, end to end

```
documents -> tokenized shards -> manifests -> mixture schedule -> packing
          -> batches -> training -> consumption ledger -> learning ledger
          -> checkpoint -> crash -> resume -> replay -> fork -> audit -> throughput
```

Every arrow is a module in `tdes/`:

| Module | Responsibility |
| --- | --- |
| `util.py` | Canonical hashing (stable across runs/restarts), config, structured logger, JSON IO |
| `tokenizer.py` | **Frozen** word-level tokenizer (stable hash) + deterministic role-segmented corpus |
| `shards.py` | Immutable tokenized shards, content hashes, manifests, **admission gate**, **eval firewall** |
| `mixture.py` | Curriculum stages → per-step lane quotas, protected floors, scarcity (repeat/synthesize) report |
| `packing.py` | Packing policies + loss masks + segment ids (block-causal) + position ids |
| `opus.py` | Data selector: accept / reject / defer + **protected-floor override** over a biased proxy |
| `model.py` | Tiny **learnable** add-k bigram model; per-token cross-entropy + perplexity |
| `ledger.py` | Append-only **consumption**, **learning**, and **OPUS** ledgers |
| `checkpoint.py` | Checkpoint = model state **+ data position** (next batch index, ledger offset, branch) |
| `trainer.py` | Deterministic plan builder, training stream, throughput, mixture compliance, audit |
| `evidence.py` | Assembles `evidence.json` / `evidence.md` from checks the run actually executed |

---

## Key design decisions

**1. Content-addressable determinism.** Every artifact (shard, packed sequence,
batch, checkpoint) is identified by a canonical hash of its content. The whole
candidate plan is a **pure function** of `(config, shards, stages)`, so replay and
fork are exact by construction — no seeds to get out of sync.

**2. A checkpoint without a data position is incomplete.** Checkpoints store
`next_batch_index` and `ledger_offset` alongside model state. Resume restores the
model *and* the data cursor, so the next batch served is provably the expected one.

**3. Batch hash is a property of data, not selection.** `batch_hash` is computed
from the packed sequences before OPUS runs. OPUS then zeroes the loss mask of
rejected sequences (they still occupy batch positions — which is why they cost
throughput). This keeps crash/resume/replay deterministic while OPUS decisions
remain a separate, audited layer.

**4. The proxy is deliberately biased.** `opus.py` under-scores Indic / agentic /
reasoning lanes (an English-heavy proxy). Without protection those lanes collapse;
the **protected-floor override** rescues them, and every rescue is logged. The demo
shows 57 protected overrides — visible proof the floor is doing work.

**5. Structure-preserving masks.** Attention boundaries and position ids reset at
the **document** level (block-causal — no cross-document leakage), while source refs
are finer (per role) so audits and mask checks are precise. Agentic tool
observations and SFT prompts are context-only (**zero loss**); pad never bears loss.

**6. Two-way ledger.** The consumption ledger records *what was served* (shards,
spans, mask hashes, lane, stage, checkpoint). The learning ledger records *what
happened* (per-lane loss, high-perplexity clusters, gradient-norm proxy, model
age), closing the loop that tells V6 what to collect, protect, repeat, or reject.

---

## How each requirement is proven (all reconstructable, none simulated)

| Requirement | How it is proven in `run_demo.py` |
| --- | --- |
| Tokenizer integrity | Two tokenizer instances produce the same hash; every manifest carries it |
| Shard immutability | Rebuilding a shard yields the same `content_hash`; a tampered token array does not |
| Manifest admission | 6/6 train shards admitted, 3/3 eval shards rejected by the gate |
| Evaluation firewall | Eval/test shards registered and **blocked**; a leak check confirms no eval shard is ever consumed |
| Packing / masks / ids | All 256 packed sequences pass structural validation; context roles + pad carry zero loss |
| Mixture + floors + OPUS | Stages validated; protected floors hold in the realized stream; OPUS accept/reject/defer/override all present |
| Consumption + learning ledgers | 32 events each; learning linked to consumption by `batch_hash`; loss drops 4.35 → 2.44 |
| Crash / resume | Crash at step 9 → restore last durable checkpoint (step 8) → next batch hash matches original; full stream identical & contiguous |
| Replay | Interval `[8,12)` rebuilt from scratch matches original batch hashes, sample ids and token spans |
| Fork | New branch from a checkpoint with an Indic-heavy mixture; prefix identical, diverges at step 26, new branch id recorded |
| Audit | Token range → influencing steps + shards reconstructed from the ledger |
| Throughput | Packing utilization ~0.98 and useful-tokens/sec recomputed identically from plan + ledger |

---

## Verification model (mirrors the assignment's three steps)

1. **Execute** — `python3 run_demo.py` regenerates `submission_artifacts/`.
2. **Verify evidence** — `evidence.json` / `evidence.md` cross-reference the generated
   manifests, ledgers, checkpoints, and `performance.json`.
3. **Inspect code** — `tests/test_invariants.py` asserts the invariants independently
   (determinism, mask correctness, firewall enforcement, exact crash-recovery,
   replay fidelity, protected floors), so the evidence reflects real behaviour.

## Notes / honest scope

- The "model" is a real, learnable add-k bigram — deterministic and serializable so
  checkpoint/resume are exact. It is a toy by design; the *data system* around it is
  the deliverable.
- The corpus is synthetic and small; Indic is deliberately under-supplied so the
  mixture compiler flags it for repetition/synthesis (mirroring Session 5).
- Throughput uses a fixed simulated per-position cost so numbers are reproducible;
  the metric that matters — **useful loss-bearing tokens per second** — is computed
  from the ledger and re-derivable.
