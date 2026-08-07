"""Evidence bundle generator: evidence.json (machine-readable) + evidence.md (human).

Every row is derived from `logger.passes` (the checks the implementation actually
ran) plus pointers to the generated artifacts. Nothing here is hardcoded — if a
check did not run or failed, it shows up as such.
"""

from __future__ import annotations

from .util import write_json

# maps the assignment's evidence rows -> the internal check keys that back them
REQUIREMENTS = [
    ("Tokenizer integrity",   "tokenizer_hash_verified",   "manifests/*.json (tokenizer_hash)"),
    ("Shard immutability",    "shards_immutable_content_hash", "manifests/*.json (content_hash)"),
    ("Manifest admission",    "manifests_validated",       "run.log admission events"),
    ("Evaluation firewall",   "eval_shard_blocked",        "ledgers/firewall.json blocked events"),
    ("Firewall no-leak",      "eval_firewall_no_leak",     "ledgers/consumption.jsonl shard_ids"),
    ("Mixture schedule",      "mixture_stages_valid",      "manifests/mixture_timeline.json"),
    ("Mixture compliance",    "protected_floor_respected", "performance.json mixture_compliance"),
    ("Packing correctness",   "packing_masks_valid",       "packed-sequence validation"),
    ("Loss/attention/pos ids","loss_mask_correctness",     "packed-sequence validation"),
    ("OPUS audit trail",      "opus_audit_trail",          "ledgers/opus.jsonl"),
    ("Consumption ledger",    "consumption_ledger_complete","ledgers/consumption.jsonl"),
    ("Learning ledger",       "learning_ledger_linked",    "ledgers/learning.jsonl"),
    ("Model learns",          "model_learns",              "ledgers/learning.jsonl"),
    ("Checkpoint saved",      "checkpoint_saved",          "checkpoints/*.json"),
    ("Crash recovery",        "resume_next_batch_matched", "run.log resume events"),
    ("No skip/repeat",        "no_skip_or_repeat",         "ledgers/consumption.jsonl"),
    ("Replay",                "replay_hash_matched",       "run.log replay events"),
    ("Fork",                  "fork_branch_diverged",      "ledgers/fork.json"),
    ("Audit",                 "audit_reconstructed",       "ledgers/audit.json"),
    ("Throughput",            "throughput_reconstructable","performance.json"),
]


def generate_evidence(out_dir: str, passes: dict, artifacts: dict) -> dict:
    rows = []
    for label, key, evidence in REQUIREMENTS:
        ran = key in passes
        result = "PASS" if passes.get(key) else ("FAIL" if ran else "NOT-RUN")
        rows.append({"requirement": label, "check_key": key,
                     "result": result, "evidence": evidence})

    all_pass = all(passes.get(k) for _, k, _ in REQUIREMENTS)
    bundle = {
        "summary": {
            "all_requirements_passed": all_pass,
            "checks_run": len(passes),
            "checks_passed": sum(1 for v in passes.values() if v),
        },
        "requirements": rows,
        "artifacts": artifacts,
        "checks": passes,
    }
    write_json(f"{out_dir}/evidence.json", bundle)

    # human-readable markdown
    lines = ["# V5 Training Data Execution System — Evidence Bundle", "",
             f"**Overall:** {'ALL PASS' if all_pass else 'SEE TABLE'}  "
             f"({bundle['summary']['checks_passed']}/{bundle['summary']['checks_run']} checks passed)",
             "", "| Requirement | Result | Evidence |", "|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['requirement']} | {r['result']} | `{r['evidence']}` |")
    lines += ["", "## Key artifacts", ""]
    for k, v in artifacts.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    with open(f"{out_dir}/evidence.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return bundle
