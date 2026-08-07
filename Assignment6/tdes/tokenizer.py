"""Frozen tokenizer + deterministic synthetic corpus generator.

The tokenizer is a word-level tokenizer over a *fixed* vocabulary, so its hash is
stable across runs (Session 2's tokenizer contract). The corpus generator emits
typed, role-segmented documents per capability lane plus held-out eval/validation
data, so downstream loss masks are meaningful and the eval firewall has something
to block.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .util import CAPABILITY_LANES, stable_hash

# Special tokens (canonical ids, Session 2 contract)
PAD, EOS, BOS, UNK = 0, 1, 2, 3
SPECIAL = {"<pad>": PAD, "<eos>": EOS, "<bos>": BOS, "<unk>": UNK}

# Role markers separate context-only spans from loss-bearing spans.
ROLE_MARKERS = ["<user>", "<plan>", "<call>", "<obs>", "<final>", "<think>", "<answer>"]

# Fixed per-lane vocabularies (kept small; frozen so tokenizer_hash is stable).
LANE_WORDS: dict[str, list[str]] = {
    "general_web": ["the", "world", "news", "today", "people", "city", "history", "weather",
                    "market", "report", "story", "event", "public", "region", "update"],
    "code": ["def", "return", "import", "class", "for", "if", "else", "value", "list",
             "func", "arg", "print", "range", "self", "None"],
    "math_science": ["theorem", "proof", "integral", "vector", "matrix", "energy", "atom",
                     "prime", "limit", "derivative", "equals", "sum", "solve", "lemma", "graph"],
    "indic": ["namaste", "bharat", "bhasha", "gyan", "vidya", "samay", "desh", "loka",
              "shanti", "sanskriti", "kavita", "ganit", "vigyan", "sahitya", "itihaas"],
    "reasoning": ["step", "because", "therefore", "assume", "consider", "compute", "verify",
                  "hence", "suppose", "conclude", "check", "thus", "given", "claim", "result"],
    "agentic": ["search", "query", "tool", "fetch", "result", "grant", "lab", "plan",
                "select", "invoke", "observe", "recover", "retry", "final", "answer"],
}


class Tokenizer:
    """Deterministic word-level tokenizer with a frozen, hashable vocabulary."""

    def __init__(self, version: str):
        self.version = version
        vocab: dict[str, int] = dict(SPECIAL)
        idx = len(vocab)
        for marker in ROLE_MARKERS:
            vocab[marker] = idx
            idx += 1
        # deterministic order: sorted lanes, then sorted words
        for lane in sorted(LANE_WORDS):
            for word in sorted(LANE_WORDS[lane]):
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def hash(self) -> str:
        # tokenizer_hash gives meaning to token ids; binds version + full vocab
        return stable_hash({"version": self.version, "vocab": self.vocab})

    def encode(self, words: list[str]) -> list[int]:
        return [self.vocab.get(w, UNK) for w in words]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.inv_vocab.get(i, "<unk>") for i in ids]


# --------------------------------------------------------------------------
# Corpus model
# --------------------------------------------------------------------------
@dataclass
class Segment:
    role: str          # e.g. "text", "user", "plan", "call", "obs", "final"
    words: list[str]
    loss: bool         # True -> loss-bearing, False -> context-only


@dataclass
class Document:
    doc_id: str
    lane: str
    kind: str          # "pretrain" | "sft" | "agentic"
    split: str         # "train" | "val" | "test" | "eval"
    source_id: str
    license_tier: str
    provenance_tier: str
    segments: list[Segment] = field(default_factory=list)


def _sample(rng: random.Random, lane: str, n: int) -> list[str]:
    words = LANE_WORDS[lane]
    return [words[rng.randrange(len(words))] for _ in range(n)]


def _make_document(rng: random.Random, lane: str, split: str, n: int) -> Document:
    doc_uid = stable_hash({"lane": lane, "split": split, "n": n, "r": rng.random()})
    src = f"src::{lane}::{split}"
    lic = "cc-by" if split == "train" else "eval-only"
    prov = "A" if lane in ("indic", "agentic") else "B"

    if lane == "agentic":
        kind = "agentic"
        segs = [
            Segment("user", _sample(rng, lane, rng.randint(3, 6)), loss=False),
            Segment("plan", _sample(rng, lane, rng.randint(3, 6)), loss=True),
            Segment("call", _sample(rng, lane, rng.randint(2, 4)), loss=True),
            Segment("obs", _sample(rng, lane, rng.randint(3, 6)), loss=False),   # tool output: NO loss
            Segment("call", _sample(rng, lane, rng.randint(2, 4)), loss=True),
            Segment("obs", _sample(rng, lane, rng.randint(3, 6)), loss=False),
            Segment("final", _sample(rng, lane, rng.randint(3, 6)), loss=True),
        ]
    elif lane == "reasoning":
        kind = "sft"
        segs = [
            Segment("user", _sample(rng, lane, rng.randint(3, 5)), loss=False),   # prompt: context
            Segment("think", _sample(rng, lane, rng.randint(6, 12)), loss=True),
            Segment("answer", _sample(rng, lane, rng.randint(2, 4)), loss=True),
        ]
    else:
        kind = "pretrain"
        segs = [Segment("text", _sample(rng, lane, n), loss=True)]  # plain LM: all real tokens
    return Document(doc_uid, lane, kind, split, src, lic, prov, segs)


def generate_corpus(seed: int) -> dict[str, list[Document]]:
    """Return documents grouped by (lane, split). Deterministic from seed."""
    rng = random.Random(seed)
    docs: dict[str, list[Document]] = {}

    # training documents per lane (indic deliberately scarce -> triggers repetition)
    per_lane_train = {
        "general_web": 24, "code": 18, "math_science": 16,
        "indic": 6, "reasoning": 10, "agentic": 8,
    }
    for lane in CAPABILITY_LANES:
        key = f"{lane}::train"
        docs[key] = [
            _make_document(rng, lane, "train", rng.randint(18, 40))
            for _ in range(per_lane_train[lane])
        ]

    # held-out validation (readable during training, never gradient-bearing)
    for lane in ("general_web", "code", "indic"):
        docs[f"{lane}::val"] = [_make_document(rng, lane, "val", 24) for _ in range(2)]

    # evaluation / test data (never-train; used to exercise the firewall)
    for bench, lane in (("MMLU", "general_web"), ("MILU", "indic"), ("BFCL", "agentic")):
        docs[f"{bench}::eval"] = [_make_document(rng, lane, "eval", 20) for _ in range(2)]

    return docs
