"""A tiny but genuinely learnable model: an add-k bigram over token ids.

It is deterministic and fully serializable, so checkpoint/resume reproduce exact
state. Loss decreases as counts accumulate, giving the learning ledger a real signal.
Per loss-bearing token it reports cross-entropy and perplexity.
"""

from __future__ import annotations

import math

from .util import Config


class BigramModel:
    def __init__(self, vocab_size: int, k: float):
        self.vocab_size = vocab_size
        self.k = k
        # counts[prev][cur]; stored sparsely as dict[str,int] for JSON round-trip
        self.counts: dict[str, int] = {}
        self.ctx_totals: dict[int, int] = {}
        self.tokens_seen = 0
        self.age_steps = 0

    # -- serialization --------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "k": self.k,
            "counts": self.counts,
            "ctx_totals": {str(k): v for k, v in self.ctx_totals.items()},
            "tokens_seen": self.tokens_seen,
            "age_steps": self.age_steps,
        }

    @classmethod
    def load_state(cls, sd: dict) -> "BigramModel":
        m = cls(sd["vocab_size"], sd["k"])
        m.counts = dict(sd["counts"])
        m.ctx_totals = {int(k): v for k, v in sd["ctx_totals"].items()}
        m.tokens_seen = sd["tokens_seen"]
        m.age_steps = sd["age_steps"]
        return m

    # -- inference / training ------------------------------------------
    def _prob(self, prev: int, cur: int) -> float:
        num = self.counts.get(f"{prev},{cur}", 0) + self.k
        den = self.ctx_totals.get(prev, 0) + self.k * self.vocab_size
        return num / den

    def token_loss(self, prev: int, cur: int) -> float:
        return -math.log(self._prob(prev, cur))

    def train_step(self, sequences: list, effective_masks: list | None = None) -> dict:
        """Compute per-loss-token cross-entropy under the CURRENT model, then update
        counts. Respects segment ids (no prediction across a segment boundary) and
        loss masks. `effective_masks` (parallel to sequences) overrides each
        sequence's loss mask — this is how OPUS rejection zeroes out a sequence's
        gradient contribution while it still occupies batch positions. Returns
        aggregate stats for this optimizer step."""
        losses: list[float] = []
        per_token: list[dict] = []
        positions_processed = 0
        for si, seq in enumerate(sequences):
            mask = seq.loss_mask if effective_masks is None else effective_masks[si]
            positions_processed += len(seq.token_ids)
            for i in range(1, len(seq.token_ids)):
                # only predict within the same segment (block-causal boundary)
                if seq.segment_ids[i] != seq.segment_ids[i - 1]:
                    continue
                if mask[i] != 1:
                    continue
                prev, cur = seq.token_ids[i - 1], seq.token_ids[i]
                loss = self.token_loss(prev, cur)
                losses.append(loss)
                per_token.append({
                    "seg": seq.segment_ids[i],
                    "pos": seq.position_ids[i],
                    "prev": prev, "cur": cur,
                    "loss": round(loss, 6),
                    "ppl": round(math.exp(loss), 4),
                    "lane": seq.lane,
                })
                # update after scoring (online learning)
                key = f"{prev},{cur}"
                self.counts[key] = self.counts.get(key, 0) + 1
                self.ctx_totals[prev] = self.ctx_totals.get(prev, 0) + 1
                self.tokens_seen += 1
        self.age_steps += 1
        avg = sum(losses) / len(losses) if losses else 0.0
        return {
            "avg_loss": round(avg, 6),
            "avg_ppl": round(math.exp(avg), 4) if losses else 0.0,
            "loss_tokens": len(losses),
            "positions_processed": positions_processed,
            "per_token": per_token,
        }
