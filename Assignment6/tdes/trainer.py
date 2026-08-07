"""The training data execution engine: builds the deterministic candidate plan,
applies OPUS, trains the model while writing ledgers and checkpoints, and runs the
crash/resume/replay/fork/audit drills that prove the stream is reconstructable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .checkpoint import save_checkpoint
from .ledger import Ledgers, summarize_opus
from .mixture import default_stages, lane_plan_for_stage, validate_stage
from .model import BigramModel
from .opus import opus_decide
from .packing import LanePool, pack_one
from .shards import Shard
from .util import CAPABILITY_LANES, PROTECTED_LANES, Config, stable_hash


# --------------------------------------------------------------------------
# Deterministic candidate plan
# --------------------------------------------------------------------------
def build_candidate_plan(cfg: Config, train_shards: list[Shard],
                         stages=None) -> list[dict]:
    """Pure, deterministic: same inputs -> identical plan (enables replay/fork)."""
    stages = stages or default_stages()
    by_lane: dict[str, list[Shard]] = {}
    for s in train_shards:
        by_lane.setdefault(s.lane, []).append(s)
    pools = {lane: LanePool(lane, by_lane[lane]) for lane in sorted(by_lane)}

    plan: list[dict] = []
    token_cursor = 0
    idx = 0
    for stage in stages:
        lane_plan = lane_plan_for_stage(stage, cfg.global_batch_seqs)
        for lanes in lane_plan:
            seqs = [pack_one(pools[lane], cfg.seq_len) for lane in lanes]
            batch_hash = stable_hash([sq.seq_id for sq in seqs] + [stage.name] + lanes)
            lane_counts: dict[str, int] = {}
            for lane in lanes:
                lane_counts[lane] = lane_counts.get(lane, 0) + 1
            plan.append({
                "candidate_index": idx,
                "stage": stage.name,
                "lanes": lanes,
                "lane_counts": lane_counts,
                "sequences": seqs,
                "batch_hash": batch_hash,
                "candidate_id": batch_hash,
                "token_start": token_cursor,
                "token_end": token_cursor + cfg.global_batch_tokens,
            })
            token_cursor += cfg.global_batch_tokens
            idx += 1
    return plan


def opus_for_plan(cfg: Config, plan: list[dict]) -> tuple[list[list[list[int]]], list[dict]]:
    """Return (per-step effective masks, flat list of decisions). Pure/deterministic."""
    step_masks: list[list[list[int]]] = []
    decisions: list[dict] = []
    for cand in plan:
        masks: list[list[int]] = []
        opus_ids: list[str] = []
        for seq in cand["sequences"]:
            cid = stable_hash([seq.seq_id, cand["candidate_index"]])
            dec = opus_decide(cid, seq.lane, cand["stage"], cand["candidate_index"],
                              seq.loss_tokens, cfg)
            decisions.append(dec)
            opus_ids.append(cid)
            if dec["status"] == "accepted":
                masks.append(list(seq.loss_mask))
            else:
                masks.append([0] * len(seq.loss_mask))  # rejected/deferred -> no gradient
        cand["opus_ids"] = opus_ids
        step_masks.append(masks)
    return step_masks, decisions


# --------------------------------------------------------------------------
# Training stream
# --------------------------------------------------------------------------
def train_stream(cfg: Config, plan: list[dict], step_masks: list, model: BigramModel,
                 ledgers: Ledgers, *, run_id: str, branch_id: str, ckpt_dir: str,
                 start_index: int, checkpoints_out: list, last_ckpt_id: str = "none",
                 crash_at: int | None = None) -> tuple[str, int, str]:
    for i in range(start_index, len(plan)):
        cand = plan[i]
        seqs = cand["sequences"]
        stats = model.train_step(seqs, step_masks[i])
        ledgers.record_consumption(run_id=run_id, branch_id=branch_id, global_step=i,
                                   checkpoint_id=last_ckpt_id, sequences=seqs,
                                   batch=cand, stage=cand["stage"], cfg=cfg)
        ledgers.record_learning(global_step=i, stage=cand["stage"], stats=stats,
                                batch=cand, model_age=model.age_steps)
        if (i + 1) % cfg.checkpoint_every == 0:
            ck = save_checkpoint(ckpt_dir, run_id=run_id, branch_id=branch_id,
                                 global_step=i + 1, model_state=model.state_dict(),
                                 rng_state={"seed": cfg.seed}, next_batch_index=i + 1,
                                 ledger_offset=len(ledgers.consumption),
                                 stage=cand["stage"])
            last_ckpt_id = ck["checkpoint_id"]
            checkpoints_out.append(ck)
        if crash_at is not None and i == crash_at:
            return ("crash", i, last_ckpt_id)
    return ("done", len(plan), last_ckpt_id)


# --------------------------------------------------------------------------
# Throughput / packing efficiency (reconstructable from plan + ledgers)
# --------------------------------------------------------------------------
def compute_throughput(cfg: Config, plan: list[dict], ledgers: Ledgers,
                       decisions: list[dict]) -> dict:
    total_positions = len(plan) * cfg.global_batch_tokens
    raw_tokens = sum(sq.real_tokens for cand in plan for sq in cand["sequences"])
    pre_opus_loss_tokens = sum(sq.loss_tokens for cand in plan for sq in cand["sequences"])
    useful_tokens = sum(ev["loss_tokens"] for ev in ledgers.learning)  # after OPUS
    time_s = total_positions * cfg.sec_per_position

    rej_by_lane: dict[str, dict[str, int]] = {}
    for d in decisions:
        lane = d["capability_lane"]
        b = rej_by_lane.setdefault(lane, {"total": 0, "rejected": 0, "deferred": 0})
        b["total"] += 1
        if d["status"] == "rejected":
            b["rejected"] += 1
        elif d["status"] == "deferred":
            b["deferred"] += 1

    return {
        "total_positions": total_positions,
        "raw_tokens": raw_tokens,
        "padding_tokens": total_positions - raw_tokens,
        "pre_opus_loss_tokens": pre_opus_loss_tokens,
        "useful_loss_tokens_after_opus": useful_tokens,
        "packing_utilization": round(raw_tokens / total_positions, 4),
        "loss_bearing_utilization": round(useful_tokens / total_positions, 4),
        "sim_time_seconds": round(time_s, 6),
        "raw_tokens_per_sec": round(raw_tokens / time_s, 1),
        "useful_tokens_per_sec": round(useful_tokens / time_s, 1),
        "accepted_tokens_per_sec": round(useful_tokens / time_s, 1),
        "rejection_rate_by_lane": {
            ln: round(v["rejected"] / v["total"], 4) for ln, v in sorted(rej_by_lane.items())
        },
    }


# --------------------------------------------------------------------------
# Mixture compliance
# --------------------------------------------------------------------------
def mixture_compliance(cfg: Config, plan: list[dict], step_masks: list) -> dict:
    """Planned vs. realized (loss-bearing, post-OPUS) lane shares + floor check."""
    planned_seqs: dict[str, int] = {}
    accepted_seqs: dict[str, int] = {}
    total_seqs = 0
    for i, cand in enumerate(plan):
        for si, seq in enumerate(cand["sequences"]):
            total_seqs += 1
            planned_seqs[seq.lane] = planned_seqs.get(seq.lane, 0) + 1
            if sum(step_masks[i][si]) > 0:      # this sequence bears loss after OPUS
                accepted_seqs[seq.lane] = accepted_seqs.get(seq.lane, 0) + 1
    total_accepted = sum(accepted_seqs.values())

    planned_share = {ln: round(planned_seqs.get(ln, 0) / total_seqs, 4) for ln in CAPABILITY_LANES}
    accepted_share = {ln: round(accepted_seqs.get(ln, 0) / total_accepted, 4)
                      for ln in CAPABILITY_LANES}

    # protected-floor invariant: min floor across stages per protected lane
    stages = default_stages()
    min_floor = {}
    for lane in PROTECTED_LANES:
        min_floor[lane] = min(s.protected_floors.get(lane, 0.0) for s in stages)
    floor_ok = {}
    for lane in PROTECTED_LANES:
        floor_ok[lane] = accepted_share.get(lane, 0.0) >= min_floor[lane] - 1e-9
    return {
        "planned_share": planned_share,
        "accepted_share": accepted_share,
        "protected_min_floor": min_floor,
        "protected_floor_respected": floor_ok,
        "all_floors_respected": all(floor_ok.values()),
    }


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------
def audit_token_range(ledgers: Ledgers, token_start: int, token_end: int) -> dict:
    hit = []
    shard_ids: set[str] = set()
    stages: set[str] = set()
    for ev in ledgers.consumption:
        if ev["token_end"] > token_start and ev["token_start"] < token_end:
            hit.append(ev["global_step"])
            shard_ids.update(ev["shard_ids"])
            stages.add(ev["curriculum_stage"])
    return {
        "query_token_range": [token_start, token_end],
        "influencing_steps": hit,
        "influencing_shards": sorted(shard_ids),
        "curriculum_stages": sorted(stages),
    }
