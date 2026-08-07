"""Immutable tokenized shards, manifests, admission gate, and the eval firewall.

A shard is sealed: its content_hash is computed from the token payload + metadata.
Any modification produces a new shard with new lineage. The admission gate enforces
the Session 3/4 contracts (tokenizer hash present, cleaning lineage known, license
safe, no eval overlap). The firewall registers eval/test shards as never-train and
blocks them from ever entering a training batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tokenizer import Document, Tokenizer, EOS
from .util import Config, hash_ints, stable_hash


@dataclass
class TokenSpan:
    """A contiguous run of tokens from one document, with per-token loss flags."""
    doc_id: str
    role: str
    token_ids: list[int]
    loss_flags: list[int]     # 1 = loss-bearing, 0 = context-only


@dataclass
class Shard:
    shard_id: str
    lane: str
    split: str
    spans: list[TokenSpan]
    manifest: dict = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return sum(len(s.token_ids) for s in self.spans)

    @property
    def loss_token_count(self) -> int:
        return sum(sum(s.loss_flags) for s in self.spans)


def _tokenize_document(tok: Tokenizer, doc: Document) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    for seg in doc.segments:
        ids = tok.encode(seg.words)
        flags = [1 if seg.loss else 0] * len(ids)
        spans.append(TokenSpan(doc.doc_id, seg.role, ids, flags))
    # EOS terminates the document; it is a boundary marker, not loss-bearing.
    spans.append(TokenSpan(doc.doc_id, "eos", [EOS], [0]))
    return spans


def build_shard(tok: Tokenizer, cfg: Config, lane: str, split: str,
                docs: list[Document]) -> Shard:
    spans: list[TokenSpan] = []
    for doc in docs:
        spans.extend(_tokenize_document(tok, doc))

    all_ids = [i for s in spans for i in s.token_ids]
    content_hash = hash_ints(all_ids)
    doc_ids = sorted({s.doc_id for s in spans})

    manifest = {
        "shard_id": "",  # filled below (depends on content hash)
        "lane": lane,
        "split": split,
        "source_ids": sorted({d.source_id for d in docs}),
        "document_ids": doc_ids,
        "tokenizer_hash": tok.hash(),
        "tokenizer_version": tok.version,
        "token_count": len(all_ids),
        "loss_token_count": sum(sum(s.loss_flags) for s in spans),
        "language": "hi" if lane == "indic" else "en",
        "script": "deva" if lane == "indic" else "latin",
        "capability_lane": lane,
        "license_tier": docs[0].license_tier if docs else "unknown",
        "provenance_tier": docs[0].provenance_tier if docs else "unknown",
        "cleaning_pipeline_hash": stable_hash(cfg.cleaning_pipeline_version),
        "dedup_status": "deduplicated",
        "contamination_status": "clean" if split == "train" else "held_out",
        "eval_overlap": (split in ("eval", "test")),
        "content_hash": content_hash,
        "parent_shard_ids": [],
    }
    shard_id = f"shard::{lane}::{split}::{stable_hash(manifest, 12)}"
    manifest["shard_id"] = shard_id
    return Shard(shard_id, lane, split, spans, manifest)


# --------------------------------------------------------------------------
# Admission gate (Session 3/4 contracts)
# --------------------------------------------------------------------------
def admission_check(manifest: dict, tok: Tokenizer) -> tuple[bool, str]:
    """Return (admitted, reason). A training shard is admitted only if it is
    clean, licensed, tokenizer-bound and free of eval overlap."""
    if not manifest.get("tokenizer_hash"):
        return False, "missing tokenizer hash"
    if manifest["tokenizer_hash"] != tok.hash():
        return False, "tokenizer hash mismatch"
    if not manifest.get("cleaning_pipeline_hash"):
        return False, "unknown cleaning lineage"
    if manifest.get("eval_overlap"):
        return False, "eval overlap"
    if manifest.get("split") != "train":
        return False, f"non-train split '{manifest.get('split')}'"
    if manifest.get("license_tier") not in ("cc-by", "cc0", "public-domain", "permissive"):
        return False, f"unsafe license '{manifest.get('license_tier')}'"
    if manifest.get("contamination_status") != "clean":
        return False, "not marked clean"
    return True, "admitted"


# --------------------------------------------------------------------------
# Eval firewall: a registry of never-train shards + content/benchmark fingerprints
# --------------------------------------------------------------------------
@dataclass
class EvalFirewall:
    content_hashes: set[str] = field(default_factory=set)
    benchmark_ids: set[str] = field(default_factory=set)
    registry: list[dict] = field(default_factory=list)
    blocked_events: list[dict] = field(default_factory=list)

    def register(self, shard: Shard, benchmark_id: str, kind: str) -> None:
        self.content_hashes.add(shard.manifest["content_hash"])
        self.benchmark_ids.add(benchmark_id)
        self.registry.append({
            "shard_id": shard.shard_id,
            "benchmark_id": benchmark_id,
            "kind": kind,                       # "validation" | "test"
            "content_hash": shard.manifest["content_hash"],
            "never_train": kind == "test",
            "read_for_eval": True,
        })

    def check_batch_shard(self, shard: Shard) -> tuple[bool, str]:
        """Return (allowed, reason). Blocks any shard whose content hash matches
        a registered eval/test fingerprint."""
        ch = shard.manifest["content_hash"]
        if ch in self.content_hashes:
            self.blocked_events.append({
                "event": "eval_shard_blocked",
                "shard_id": shard.shard_id,
                "content_hash": ch,
                "reason": "matches registered eval/test fingerprint",
            })
            return False, "eval overlap fingerprint"
        return True, "ok"
