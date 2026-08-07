"""Append-only ledgers: consumption, learning, and OPUS decisions.

The consumption ledger is the run's memory — every served optimizer step records the
shard ids, token spans, mask hashes, lane, stage and checkpoint so any step can be
reconstructed, replayed, or audited. The learning ledger attaches outcomes (loss,
perplexity, gradient proxy) back to the data, closing the two-way loop for V6.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .util import canonical_json, stable_hash, write_jsonl


@dataclass
class Ledgers:
    consumption: list[dict] = field(default_factory=list)
    learning: list[dict] = field(default_factory=list)
    opus: list[dict] = field(default_factory=list)

    # -- consumption ----------------------------------------------------
    def record_consumption(self, *, run_id: str, branch_id: str, global_step: int,
                           checkpoint_id: str, sequences: list, batch: dict,
                           stage: str, cfg) -> dict:
        packed_ids, shard_ids, span_ids, loss_hashes = [], set(), [], []
        for rank, seq in enumerate(sequences):
            packed_ids.append(seq.seq_id)
            loss_hashes.append(stable_hash(seq.loss_mask))
            for ref in seq.source_refs:
                shard_ids.add(ref["shard_id"])
                span_ids.append(f"{ref['shard_id']}:{ref['doc_id']}:{ref['start']}+{ref['length']}")
        event = {
            "run_id": run_id,
            "branch_id": branch_id,
            "global_step": global_step,
            "checkpoint_id": checkpoint_id,
            "microbatch_layout": {"n_gpus": cfg.n_gpus, "microbatch": cfg.microbatch,
                                   "grad_accum": cfg.grad_accum},
            "packed_sample_ids": packed_ids,
            "shard_ids": sorted(shard_ids),
            "token_span_ids": span_ids,
            "loss_mask_hash": stable_hash(loss_hashes),
            "attention_policy": sequences[0].attention_policy if sequences else "none",
            "position_policy": "reset_per_segment",
            "mixture_lane_counts": batch["lane_counts"],
            "curriculum_stage": stage,
            "tokenizer_version": cfg.tokenizer_version,
            "dataloader_version": cfg.dataloader_version,
            "opus_decision_id": batch["candidate_id"],
            "batch_hash": batch["batch_hash"],
            "token_start": batch["token_start"],
            "token_end": batch["token_end"],
        }
        self.consumption.append(event)
        return event

    # -- learning -------------------------------------------------------
    def record_learning(self, *, global_step: int, stage: str, stats: dict,
                        batch: dict, model_age: int) -> dict:
        # aggregate per-lane and per-token-cluster signal
        by_lane: dict[str, list[float]] = {}
        for t in stats["per_token"]:
            by_lane.setdefault(t["lane"], []).append(t["loss"])
        lane_avg = {ln: round(sum(v) / len(v), 6) for ln, v in by_lane.items()}
        high_ppl = sorted(stats["per_token"], key=lambda t: -t["ppl"])[:5]
        event = {
            "global_step": global_step,
            "curriculum_stage": stage,
            "batch_hash": batch["batch_hash"],
            "avg_loss": stats["avg_loss"],
            "avg_ppl": stats["avg_ppl"],
            "loss_tokens": stats["loss_tokens"],
            "per_lane_avg_loss": lane_avg,
            "high_perplexity_tokens": [
                {"prev": t["prev"], "cur": t["cur"], "ppl": t["ppl"], "lane": t["lane"]}
                for t in high_ppl
            ],
            "gradient_norm_proxy": round(stats["avg_loss"] * (1 + stats["loss_tokens"] / 100.0), 6),
            "model_age_steps": model_age,
        }
        self.learning.append(event)
        return event

    # -- opus -----------------------------------------------------------
    def record_opus(self, decision: dict) -> None:
        self.opus.append(decision)

    # -- persistence ----------------------------------------------------
    def truncate_consumption_to(self, offset: int) -> None:
        """Roll the ledger back to a checkpoint offset (crash recovery)."""
        self.consumption = self.consumption[:offset]

    def flush(self, ledger_dir: str) -> None:
        write_jsonl(f"{ledger_dir}/consumption.jsonl", self.consumption)
        write_jsonl(f"{ledger_dir}/learning.jsonl", self.learning)
        write_jsonl(f"{ledger_dir}/opus.jsonl", self.opus)


def summarize_opus(opus: list[dict]) -> dict:
    counts = {"accepted": 0, "rejected": 0, "deferred": 0, "protected_override": 0}
    for d in opus:
        counts[d["status"]] = counts.get(d["status"], 0) + 1
        if d.get("protected_floor_override"):
            counts["protected_override"] += 1
    return counts
