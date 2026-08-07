"""Foundational utilities: canonical hashing, config constants, structured logging, JSON IO.

Everything downstream depends on *stable* hashing so that manifests, batch ids,
ledger events and replay comparisons are byte-for-byte reproducible across runs
and across process restarts. Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------
# Canonical hashing / serialization
# --------------------------------------------------------------------------
def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(obj: Any, length: int = 16) -> str:
    """Content-addressable hash of any JSON-able object."""
    payload = canonical_json(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def hash_ints(ints: list[int], length: int = 16) -> str:
    """Hash a token/id array compactly and deterministically."""
    h = hashlib.sha256()
    h.update(b"|".join(str(i).encode() for i in ints))
    return h.hexdigest()[:length]


# --------------------------------------------------------------------------
# Global configuration (frozen for a run; part of the experiment definition)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    seed: int = 20250607
    tokenizer_version: str = "v5-tok-1"
    cleaning_pipeline_version: str = "clean-v4.2"
    dataloader_version: str = "loader-v6.0"
    proxy_version: str = "opus-proxy-v5"

    seq_len: int = 64
    n_gpus: int = 2
    microbatch: int = 2           # sequences per GPU per forward
    grad_accum: int = 2           # accumulation steps per optimizer update
    checkpoint_every: int = 4     # optimizer steps between checkpoints

    opus_accept_fraction: float = 0.6   # top fraction retained by the selector
    opus_defer_band: float = 0.15       # band just below threshold -> deferred

    # bigram model smoothing
    model_smoothing_k: float = 0.5

    # simulated per-position service cost (seconds) for reproducible throughput
    sec_per_position: float = 1.0e-7

    @property
    def global_batch_seqs(self) -> int:
        return self.n_gpus * self.microbatch * self.grad_accum

    @property
    def global_batch_tokens(self) -> int:
        return self.global_batch_seqs * self.seq_len


CAPABILITY_LANES = [
    "general_web",
    "code",
    "math_science",
    "indic",
    "reasoning",
    "agentic",
]

# lanes whose scarce, high-value data is protected from the OPUS selector
PROTECTED_LANES = {"indic", "agentic", "reasoning"}


# --------------------------------------------------------------------------
# Structured logging -> run.log + in-memory event list
# --------------------------------------------------------------------------
@dataclass
class RunLogger:
    path: str
    lines: list[str] = field(default_factory=list)
    passes: dict[str, bool] = field(default_factory=dict)

    def _emit(self, text: str) -> None:
        self.lines.append(text)
        print(text)

    def info(self, msg: str) -> None:
        self._emit(f"[INFO] {msg}")

    def step(self, msg: str) -> None:
        self._emit(f"[STEP] {msg}")

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        tag = "PASS" if ok else "FAIL"
        self.passes[name] = ok
        suffix = f"  ({detail})" if detail else ""
        self._emit(f"[{tag}] {name}{suffix}")
        return ok

    def flush(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lines) + "\n")


# --------------------------------------------------------------------------
# JSON file IO helpers
# --------------------------------------------------------------------------
def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_jsonl(path: str, rows: list[Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(canonical_json(row) + "\n")


def read_jsonl(path: str) -> list[Any]:
    out: list[Any] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
