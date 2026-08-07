"""Packing policies that turn shard spans into fixed-length training sequences.

Every packed sequence carries the *training meaning* of its tokens: a loss mask,
segment ids (block-causal attention => no cross-document leakage), position ids
(reset per segment), and source refs (shard/doc/role) for audit & reconstruction.

Two policies are implemented and picked by data type:
  - concat_chop         : pretraining plain text (documents chopped across windows)
  - structure_preserving: SFT / agentic / reasoning (whole documents, tail padded)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .shards import Shard
from .tokenizer import PAD, EOS
from .util import hash_ints, stable_hash


@dataclass
class TokenCell:
    tid: int
    loss: int
    doc_id: str
    shard_id: str
    role: str


@dataclass
class PackedSequence:
    token_ids: list[int]
    loss_mask: list[int]
    segment_ids: list[int]
    position_ids: list[int]
    attention_policy: str            # "block_causal"
    source_refs: list[dict]          # [{doc_id, shard_id, role, start, length}]
    lane: str
    policy: str

    @property
    def seq_id(self) -> str:
        return hash_ints(self.token_ids + self.loss_mask + self.segment_ids, 16)

    @property
    def real_tokens(self) -> int:
        return sum(1 for t in self.token_ids if t != PAD)

    @property
    def loss_tokens(self) -> int:
        return sum(self.loss_mask)

    def validate(self, seq_len: int) -> list[str]:
        errs = []
        n = seq_len
        if not (len(self.token_ids) == len(self.loss_mask) ==
                len(self.segment_ids) == len(self.position_ids) == n):
            errs.append("length mismatch across parallel arrays")
        # position ids reset per segment and are contiguous within a segment
        seg_start: dict[int, int] = {}
        for i, (seg, pos) in enumerate(zip(self.segment_ids, self.position_ids)):
            if seg not in seg_start:
                seg_start[seg] = pos
                if pos != 0:
                    errs.append(f"segment {seg} first position {pos} != 0")
            expected = seg_start[seg] + (pos - seg_start[seg])
            if pos < 0 or pos >= n:
                errs.append(f"position {pos} out of range")
        # pad tokens must never bear loss
        for t, m in zip(self.token_ids, self.loss_mask):
            if t == PAD and m != 0:
                errs.append("pad token carries loss")
        return errs


class LanePool:
    """Ordered document stream for one lane, with wraparound repetition counting."""

    def __init__(self, lane: str, shards: list[Shard]):
        self.lane = lane
        self.docs: list[list[TokenCell]] = []
        # group spans into whole documents (preserving order, including EOS span)
        for shard in shards:
            by_doc: dict[str, list[TokenCell]] = {}
            order: list[str] = []
            for span in shard.spans:
                if span.doc_id not in by_doc:
                    by_doc[span.doc_id] = []
                    order.append(span.doc_id)
                for tid, lf in zip(span.token_ids, span.loss_flags):
                    by_doc[span.doc_id].append(
                        TokenCell(tid, lf, span.doc_id, shard.shard_id, span.role))
            for doc_id in order:
                self.docs.append(by_doc[doc_id])
        self.cursor = 0
        self.passes = 0                 # how many full wraps (repetition budget)
        self.queue: list[TokenCell] = []  # rolling buffer for concat_chop

    def next_doc(self) -> list[TokenCell]:
        if not self.docs:
            raise RuntimeError(f"lane '{self.lane}' has no documents")
        doc = self.docs[self.cursor]
        self.cursor += 1
        if self.cursor >= len(self.docs):
            self.cursor = 0
            self.passes += 1
        return doc


def _finalize(cells: list[TokenCell], seq_len: int, lane: str, policy: str) -> PackedSequence:
    token_ids, loss_mask, segment_ids, position_ids = [], [], [], []
    source_refs: list[dict] = []
    seg = -1
    prev_doc = None
    prev_role_key = None
    pos = 0
    cur_ref = None
    for c in cells:
        # attention boundary + position reset happen at the DOCUMENT level
        if c.doc_id != prev_doc:
            seg += 1
            pos = 0
            prev_doc = c.doc_id
        # source refs are finer (per role) for precise audit + mask checks
        role_key = (c.doc_id, c.role)
        if role_key != prev_role_key:
            cur_ref = {"doc_id": c.doc_id, "shard_id": c.shard_id,
                       "role": c.role, "start": len(token_ids), "length": 0}
            source_refs.append(cur_ref)
            prev_role_key = role_key
        token_ids.append(c.tid)
        loss_mask.append(c.loss)
        segment_ids.append(seg)
        position_ids.append(pos)
        pos += 1
        cur_ref["length"] += 1
    # pad tail to seq_len (padding segment = -1, no loss, position 0)
    while len(token_ids) < seq_len:
        token_ids.append(PAD)
        loss_mask.append(0)
        segment_ids.append(-1)
        position_ids.append(0)
    return PackedSequence(token_ids[:seq_len], loss_mask[:seq_len],
                          segment_ids[:seq_len], position_ids[:seq_len],
                          "block_causal", source_refs, lane, policy)


def pack_concat_chop(pool: LanePool, seq_len: int) -> PackedSequence:
    """Fill exactly seq_len by chopping documents across window boundaries."""
    while len(pool.queue) < seq_len:
        pool.queue.extend(pool.next_doc())
    window = pool.queue[:seq_len]
    pool.queue = pool.queue[seq_len:]
    return _finalize(window, seq_len, pool.lane, "concat_chop")


def pack_structure_preserving(pool: LanePool, seq_len: int) -> PackedSequence:
    """Pack whole documents only; pad the tail. Truncate a lone oversized doc."""
    cells: list[TokenCell] = []
    # always take at least one document
    doc = pool.next_doc()
    if len(doc) > seq_len:
        doc = doc[:seq_len]
    cells.extend(doc)
    # add further whole docs while they fit
    while True:
        nxt = pool.docs[pool.cursor]
        if len(cells) + len(nxt) > seq_len:
            break
        cells.extend(pool.next_doc())
    return _finalize(cells, seq_len, pool.lane, "structure_preserving")


def policy_for_lane(lane: str) -> str:
    if lane in ("agentic", "reasoning"):
        return "structure_preserving"
    return "concat_chop"


def pack_one(pool: LanePool, seq_len: int) -> PackedSequence:
    policy = policy_for_lane(pool.lane)
    if policy == "structure_preserving":
        return pack_structure_preserving(pool, seq_len)
    return pack_concat_chop(pool, seq_len)
