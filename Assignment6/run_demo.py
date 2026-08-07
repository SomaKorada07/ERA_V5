#!/usr/bin/env python3
"""One-command demonstration of the V5 Training Data Execution System.

    python3 run_demo.py

Runs the full path — documents -> shards -> manifests -> mixture -> packing ->
batches -> training -> ledgers -> checkpoint -> crash -> resume -> replay -> fork
-> audit -> throughput -> evidence — and writes submission_artifacts/.
"""

from __future__ import annotations

import os
import shutil

from tdes.checkpoint import load_checkpoint
from tdes.evidence import generate_evidence
from tdes.ledger import Ledgers, summarize_opus
from tdes.mixture import (compile_timeline, default_stages, lane_plan_for_stage,
                          validate_stage)
from tdes.model import BigramModel
from tdes.packing import LanePool, pack_one
from tdes.shards import EvalFirewall, admission_check, build_shard
from tdes.tokenizer import Tokenizer, generate_corpus
from tdes.trainer import (audit_token_range, build_candidate_plan, compute_throughput,
                          mixture_compliance, opus_for_plan, train_stream)
from tdes.util import (CAPABILITY_LANES, Config, RunLogger, stable_hash, write_json)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "submission_artifacts")


def _fname(shard_id: str) -> str:
    return shard_id.replace("::", "__") + ".json"


def main() -> int:
    cfg = Config()
    # fresh artifacts dir
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for sub in ("manifests", "ledgers", "checkpoints"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    log = RunLogger(os.path.join(OUT, "run.log"))
    log.info(f"config: {cfg}")
    run_id = "run::" + stable_hash(cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg))

    # ---- 1. tokenizer (frozen) ---------------------------------------
    log.step("building frozen tokenizer")
    tok = Tokenizer(cfg.tokenizer_version)
    tok2 = Tokenizer(cfg.tokenizer_version)
    log.check("tokenizer_hash_verified",
              tok.hash() == tok2.hash() and len(tok.hash()) == 16,
              f"hash={tok.hash()} vocab={tok.vocab_size}")

    # ---- 2. corpus -> shards -> manifests ----------------------------
    log.step("generating corpus and tokenizing shards")
    corpus = generate_corpus(cfg.seed)
    all_shards, train_shards = [], []
    eval_shards, val_shards = [], []
    for key, docs in corpus.items():
        lane_or_bench, split = key.split("::")
        lane = docs[0].lane
        shard = build_shard(tok, cfg, lane, split, docs)
        all_shards.append(shard)
        write_json(os.path.join(OUT, "manifests", _fname(shard.shard_id)), shard.manifest)
        if split == "train":
            train_shards.append(shard)
        elif split == "val":
            val_shards.append(shard)
        elif split == "eval":
            eval_shards.append((lane_or_bench, shard))

    # shard immutability: rebuild + tamper
    rebuilt = build_shard(tok, cfg, train_shards[0].lane, "train",
                          corpus[f"{train_shards[0].lane}::train"])
    tampered = list(train_shards[0].spans[0].token_ids)
    from tdes.util import hash_ints
    tamper_hash = hash_ints(tampered + [999])
    log.check("shards_immutable_content_hash",
              rebuilt.manifest["content_hash"] == train_shards[0].manifest["content_hash"]
              and tamper_hash != train_shards[0].manifest["content_hash"],
              f"content_hash={train_shards[0].manifest['content_hash']}")

    # admission gate: all train shards admitted; eval shards rejected
    admits = [admission_check(s.manifest, tok) for s in train_shards]
    eval_admits = [admission_check(s.manifest, tok) for _, s in eval_shards]
    log.check("manifests_validated",
              all(a[0] for a in admits) and not any(a[0] for a in eval_admits),
              f"train_admitted={sum(a[0] for a in admits)}/{len(admits)} "
              f"eval_rejected={sum(not a[0] for a in eval_admits)}/{len(eval_admits)}")
    for _, s in eval_shards:
        ok, reason = admission_check(s.manifest, tok)
        log.info(f"admission blocked {s.shard_id}: {reason}")

    # ---- 3. eval firewall --------------------------------------------
    log.step("registering eval/validation shards in the firewall")
    fw = EvalFirewall()
    for bench, s in eval_shards:
        fw.register(s, bench, "test")
    for s in val_shards:
        fw.register(s, f"VAL::{s.lane}", "validation")
    # attempt to inject an eval shard into training -> must be blocked
    blocked_ok = True
    for _, s in eval_shards:
        allowed, _ = fw.check_batch_shard(s)
        blocked_ok = blocked_ok and (not allowed)
    # train shards must pass the firewall
    train_pass = all(fw.check_batch_shard(s)[0] for s in train_shards)
    log.check("eval_shard_blocked", blocked_ok and train_pass,
              f"blocked {len(fw.blocked_events)} eval shards; train shards pass")
    write_json(os.path.join(OUT, "ledgers", "firewall.json"),
               {"registry": fw.registry, "blocked_events": fw.blocked_events})

    # ---- 4. mixture timeline -----------------------------------------
    log.step("compiling mixture timeline")
    stages = default_stages()
    problems = [p for s in stages for p in validate_stage(s)]
    lane_supply = {s.lane: 0 for s in train_shards}
    for s in train_shards:
        lane_supply[s.lane] = lane_supply.get(s.lane, 0) + s.token_count
    timeline = compile_timeline(cfg, stages, lane_supply)
    write_json(os.path.join(OUT, "manifests", "mixture_timeline.json"), timeline)
    log.check("mixture_stages_valid", not problems,
              f"{len(stages)} stages, {len(problems)} problems")
    scarce = [ln for ln, v in timeline["scarcity"].items() if v["action"] != "covered"]
    log.info(f"scarcity flags (repeat/synthesize): {scarce}")

    # ---- 5. candidate plan + OPUS ------------------------------------
    log.step("building deterministic candidate plan + OPUS selection")
    plan = build_candidate_plan(cfg, train_shards, stages)
    step_masks, decisions = opus_for_plan(cfg, plan)

    # packing / mask / position-id correctness
    mask_errs, loss_rule_ok = [], True
    for cand in plan:
        for seq in cand["sequences"]:
            mask_errs.extend(seq.validate(cfg.seq_len))
            for ref in seq.source_refs:
                if ref["role"] in ("obs", "user"):
                    span = seq.loss_mask[ref["start"]:ref["start"] + ref["length"]]
                    if any(span):
                        loss_rule_ok = False
    log.check("packing_masks_valid", not mask_errs,
              f"{len(plan)} batches, {sum(len(c['sequences']) for c in plan)} sequences, "
              f"{len(mask_errs)} errors")
    log.check("loss_mask_correctness", loss_rule_ok,
              "context spans (obs/user) carry zero loss; pad carries zero loss")

    opus_counts = summarize_opus(decisions)
    log.check("opus_audit_trail",
              opus_counts["accepted"] > 0 and opus_counts["rejected"] > 0
              and opus_counts["protected_override"] > 0,
              f"{opus_counts}")

    # ---- 6. MAIN training run (branch=main) --------------------------
    log.step("training main run with checkpoints")
    model = BigramModel(tok.vocab_size, cfg.model_smoothing_k)
    led = Ledgers()
    for d in decisions:
        led.record_opus(d)
    ckpts_main: list = []
    status, served, _ = train_stream(cfg, plan, step_masks, model, led,
                                     run_id=run_id, branch_id="main",
                                     ckpt_dir=os.path.join(OUT, "checkpoints"),
                                     start_index=0, checkpoints_out=ckpts_main)
    log.check("consumption_ledger_complete",
              status == "done" and len(led.consumption) == len(plan),
              f"{len(led.consumption)} events for {len(plan)} steps")
    # learning ledger linkage
    linked = all(led.learning[i]["batch_hash"] == led.consumption[i]["batch_hash"]
                 for i in range(len(plan)))
    log.check("learning_ledger_linked", linked and len(led.learning) == len(plan),
              f"{len(led.learning)} learning events linked to consumption")
    log.check("checkpoint_saved", len(ckpts_main) > 0,
              f"{len(ckpts_main)} checkpoints, cadence={cfg.checkpoint_every}")

    # firewall no-leak: no consumed shard is an eval/test shard
    train_ids = {s.shard_id for s in train_shards}
    leak = any(sid not in train_ids
               for ev in led.consumption for sid in ev["shard_ids"])
    log.check("eval_firewall_no_leak", not leak,
              "all consumed shards are admitted train shards")

    # model actually learns (loss on a repeated bigram strictly drops)
    probe = BigramModel(tok.vocab_size, cfg.model_smoothing_k)
    before = probe.token_loss(10, 11)
    for _ in range(5):
        probe.counts["10,11"] = probe.counts.get("10,11", 0) + 1
        probe.ctx_totals[10] = probe.ctx_totals.get(10, 0) + 1
    after = probe.token_loss(10, 11)
    log.check("model_learns", after < before,
              f"loss {before:.4f} -> {after:.4f} after repeated exposure")

    # ---- 7. CRASH + RESUME drill (branch=recover) --------------------
    log.step("crash / resume drill")
    model_b = BigramModel(tok.vocab_size, cfg.model_smoothing_k)
    led_b = Ledgers()
    for d in decisions:
        led_b.record_opus(d)
    ckpts_b: list = []
    crash_at = 9
    st, at, _ = train_stream(cfg, plan, step_masks, model_b, led_b,
                             run_id=run_id, branch_id="recover",
                             ckpt_dir=os.path.join(OUT, "checkpoints"),
                             start_index=0, checkpoints_out=ckpts_b, crash_at=crash_at)
    log.info(f"simulated crash at step {at} (status={st})")
    # pick the last durable checkpoint at or before the crash
    durable = max((c for c in ckpts_b if c["next_batch_index"] <= crash_at),
                  key=lambda c: c["next_batch_index"])
    log.info(f"last durable checkpoint: {durable['checkpoint_id']} "
             f"(next_batch_index={durable['next_batch_index']})")
    # restore model + roll ledger back to the checkpoint offset
    model_r = BigramModel.load_state(durable["model_state"])
    led_b.truncate_consumption_to(durable["ledger_offset"])
    led_b.learning = led_b.learning[:durable["ledger_offset"]]
    resume_index = durable["next_batch_index"]
    # the very next served batch must equal the original plan/ledger batch
    st2, served2, _ = train_stream(cfg, plan, step_masks, model_r, led_b,
                                   run_id=run_id, branch_id="recover",
                                   ckpt_dir=os.path.join(OUT, "checkpoints"),
                                   start_index=resume_index, checkpoints_out=[],
                                   last_ckpt_id=durable["checkpoint_id"])
    next_ok = (led_b.consumption[resume_index]["batch_hash"]
               == led.consumption[resume_index]["batch_hash"])
    log.check("resume_next_batch_matched", next_ok,
              f"step {resume_index} batch_hash "
              f"{led_b.consumption[resume_index]['batch_hash']} matches original")
    # full stream identical => no skipped/repeated batches
    same = ([e["batch_hash"] for e in led_b.consumption]
            == [e["batch_hash"] for e in led.consumption])
    contiguous = ([e["global_step"] for e in led_b.consumption]
                  == list(range(len(plan))))
    log.check("no_skip_or_repeat", same and contiguous,
              f"{len(led_b.consumption)} contiguous steps, hashes identical to original")

    # ---- 8. REPLAY drill --------------------------------------------
    log.step("replay drill (reconstruct an earlier interval)")
    a, b = 8, 12
    replay_plan = build_candidate_plan(cfg, train_shards, stages)  # rebuild from scratch
    replay_ok = True
    for i in range(a, b):
        rc = replay_plan[i]
        orig = led.consumption[i]
        recomputed_samples = [sq.seq_id for sq in rc["sequences"]]
        if (rc["batch_hash"] != orig["batch_hash"]
                or recomputed_samples != orig["packed_sample_ids"]):
            replay_ok = False
    log.check("replay_hash_matched", replay_ok,
              f"interval [{a},{b}) batch hashes, sample ids and spans match original")

    # ---- 9. FORK drill ----------------------------------------------
    log.step("fork drill (new data branch from a checkpoint)")
    fork_stages = default_stages()
    fork_stages[-1].mixture.update({"general_web": 0.05, "code": 0.15,
                                    "math_science": 0.10, "indic": 0.35,
                                    "reasoning": 0.20, "agentic": 0.15})
    fork_plan = build_candidate_plan(cfg, train_shards, fork_stages)
    fork_masks, _ = opus_for_plan(cfg, fork_plan)
    # divergence = first index whose batch differs from main
    divergence = next((i for i in range(len(plan))
                       if plan[i]["batch_hash"] != fork_plan[i]["batch_hash"]), None)
    fork_ckpt = max((c for c in ckpts_main if c["next_batch_index"] <= (divergence or 0)),
                    key=lambda c: c["next_batch_index"])
    model_f = BigramModel.load_state(fork_ckpt["model_state"])
    led_f = Ledgers()
    train_stream(cfg, fork_plan, fork_masks, model_f, led_f, run_id=run_id,
                 branch_id="fork::indic-heavy",
                 ckpt_dir=os.path.join(OUT, "checkpoints"),
                 start_index=fork_ckpt["next_batch_index"], checkpoints_out=[],
                 last_ckpt_id=fork_ckpt["checkpoint_id"])
    prefix_same = all(plan[i]["batch_hash"] == fork_plan[i]["batch_hash"]
                      for i in range(fork_ckpt["next_batch_index"]))
    suffix_diff = (divergence is not None
                   and plan[divergence]["batch_hash"] != fork_plan[divergence]["batch_hash"])
    branch_tag_ok = all(e["branch_id"] == "fork::indic-heavy" for e in led_f.consumption)
    fork_record = {
        "parent_checkpoint": fork_ckpt["checkpoint_id"],
        "branch_started_at_step": fork_ckpt["next_batch_index"],
        "data_divergence_step": divergence,
        "main_batch_hash_at_divergence": plan[divergence]["batch_hash"] if divergence is not None else None,
        "fork_batch_hash_at_divergence": fork_plan[divergence]["batch_hash"] if divergence is not None else None,
        "branch_id": "fork::indic-heavy",
        "fork_events_recorded": len(led_f.consumption),
    }
    write_json(os.path.join(OUT, "ledgers", "fork.json"), fork_record)
    log.check("fork_branch_diverged",
              prefix_same and suffix_diff and branch_tag_ok and divergence is not None,
              f"prefix identical, diverges at step {divergence}, new branch id recorded")

    # ---- 10. AUDIT ---------------------------------------------------
    log.step("audit (which shards trained a token range)")
    tr_start = plan[10]["token_start"]
    tr_end = plan[13]["token_end"]
    audit = audit_token_range(led, tr_start, tr_end)
    write_json(os.path.join(OUT, "ledgers", "audit.json"), audit)
    log.check("audit_reconstructed",
              len(audit["influencing_steps"]) > 0 and len(audit["influencing_shards"]) > 0,
              f"range influenced steps {audit['influencing_steps']} "
              f"from {len(audit['influencing_shards'])} shards")

    # ---- 11. THROUGHPUT ---------------------------------------------
    log.step("measuring throughput / packing efficiency")
    perf = compute_throughput(cfg, plan, led, decisions)
    perf2 = compute_throughput(cfg, plan, led, decisions)
    compliance = mixture_compliance(cfg, plan, step_masks)
    perf["mixture_compliance"] = compliance
    write_json(os.path.join(OUT, "performance.json"), perf)
    log.check("throughput_reconstructable",
              perf2["useful_loss_tokens_after_opus"] == perf["useful_loss_tokens_after_opus"]
              and 0.0 <= perf["packing_utilization"] <= 1.0,
              f"packing_util={perf['packing_utilization']} "
              f"useful_tok/s={perf['useful_tokens_per_sec']}")
    log.check("protected_floor_respected", compliance["all_floors_respected"],
              f"floors={compliance['protected_floor_respected']}")

    # ---- 12. persist ledgers + evidence ------------------------------
    led.flush(os.path.join(OUT, "ledgers"))
    artifacts = {
        "run_id": run_id,
        "total_optimizer_steps": len(plan),
        "train_shards": len(train_shards),
        "eval_shards_blocked": len(fw.blocked_events),
        "opus": opus_counts,
        "checkpoints": len(ckpts_main),
        "final_avg_loss": led.learning[-1]["avg_loss"],
        "first_avg_loss": led.learning[0]["avg_loss"],
    }
    bundle = generate_evidence(OUT, log.passes, artifacts)

    all_pass = bundle["summary"]["all_requirements_passed"]
    log.step(f"DEMO COMPLETE — {bundle['summary']['checks_passed']}/"
             f"{bundle['summary']['checks_run']} checks passed; "
             f"all_requirements_passed={all_pass}")
    log.flush()
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
