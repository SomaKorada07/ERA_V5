# V5 Training Data Execution System — Evidence Bundle

**Overall:** ALL PASS  (20/20 checks passed)

| Requirement | Result | Evidence |
|---|---|---|
| Tokenizer integrity | PASS | `manifests/*.json (tokenizer_hash)` |
| Shard immutability | PASS | `manifests/*.json (content_hash)` |
| Manifest admission | PASS | `run.log admission events` |
| Evaluation firewall | PASS | `ledgers/firewall.json blocked events` |
| Firewall no-leak | PASS | `ledgers/consumption.jsonl shard_ids` |
| Mixture schedule | PASS | `manifests/mixture_timeline.json` |
| Mixture compliance | PASS | `performance.json mixture_compliance` |
| Packing correctness | PASS | `packed-sequence validation` |
| Loss/attention/pos ids | PASS | `packed-sequence validation` |
| OPUS audit trail | PASS | `ledgers/opus.jsonl` |
| Consumption ledger | PASS | `ledgers/consumption.jsonl` |
| Learning ledger | PASS | `ledgers/learning.jsonl` |
| Model learns | PASS | `ledgers/learning.jsonl` |
| Checkpoint saved | PASS | `checkpoints/*.json` |
| Crash recovery | PASS | `run.log resume events` |
| No skip/repeat | PASS | `ledgers/consumption.jsonl` |
| Replay | PASS | `run.log replay events` |
| Fork | PASS | `ledgers/fork.json` |
| Audit | PASS | `ledgers/audit.json` |
| Throughput | PASS | `performance.json` |

## Key artifacts

- **run_id**: run::42ce2069111e48a8
- **total_optimizer_steps**: 32
- **train_shards**: 6
- **eval_shards_blocked**: 3
- **opus**: {'accepted': 203, 'rejected': 20, 'deferred': 33, 'protected_override': 57}
- **checkpoints**: 8
- **final_avg_loss**: 2.441041
- **first_avg_loss**: 4.349766
