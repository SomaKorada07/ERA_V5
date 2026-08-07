"""Checkpointing: binds model state to data state.

A checkpoint is incomplete without a data position, so it stores model weights,
scheduler/RNG state, the dataloader offset (next batch index), the ledger offset,
and the branch id. Resume/replay/fork all key off these fields.
"""

from __future__ import annotations

from .util import stable_hash, write_json, read_json


def save_checkpoint(ckpt_dir: str, *, run_id: str, branch_id: str, global_step: int,
                    model_state: dict, rng_state: dict, next_batch_index: int,
                    ledger_offset: int, stage: str) -> dict:
    body = {
        "run_id": run_id,
        "branch_id": branch_id,
        "global_step": global_step,
        "curriculum_stage": stage,
        "model_state": model_state,
        "rng_state": rng_state,
        "next_batch_index": next_batch_index,   # dataloader position
        "ledger_offset": ledger_offset,         # consumption ledger length
    }
    ckpt_id = f"ckpt::{branch_id}::step{global_step}::{stable_hash(body, 12)}"
    body["checkpoint_id"] = ckpt_id
    write_json(f"{ckpt_dir}/{ckpt_id.replace('::', '__')}.json", body)
    return body


def load_checkpoint(path: str) -> dict:
    return read_json(path)
