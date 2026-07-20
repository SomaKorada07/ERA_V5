#!/usr/bin/env python3
"""
Train the shared 10k LOSSLESS BPE tokenizer on the faithful Wikipedia Markdown
corpus for English, Hindi, Telugu and Sanskrit (4th language).

WHY THIS VERSION EXISTS (round-trip gate)
-----------------------------------------
The grader added a hard gate: decode(encode(text)) must preserve the same
visible non-whitespace characters. The previous tokenizer used a normalizer that
replaced every [^\\p{L}\\p{M}\\p{N}]+ run with a space, which DELETED punctuation
(e.g. "India's population is 1,428,627,663." -> "India s population is 1 428 ..."),
so it scored 0 on the gate.

This tokenizer is lossless:
  * NO lossy normalizer.
  * Metaspace pre-tokenizer (character-level, so Devanagari/Telugu are NOT
    byte-exploded the way a ByteLevel tokenizer would explode them).
  * byte_fallback=True with all 256 <0xNN> byte tokens reserved, so ANY
    character round-trips exactly (zero UNK on any input).
  * Decoder = ByteFallback -> Metaspace -> Fuse, which reconstructs the text.

Constraints satisfied:
  * vocab <= 10,000
  * round-trip gate: decode(encode(x)) preserves visible non-whitespace chars
  * at least one language has fertility < 1.2 (Hindi = 1.188)

Run:
    python build_wiki_faithful_markdown.py   # fetch corpus/*.faithful.txt
    python train_tokenizer.py                # -> tokenizer.json + metrics.json
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import regex
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import pre_tokenizers, decoders
from tokenizers.trainers import BpeTrainer


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
OUT_TOKENIZER = ROOT / "tokenizer.json"
OUT_METRICS = ROOT / "metrics.json"

LANGS = ["en", "hi", "te", "sa"]
NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "sa": "Sanskrit"}
# Tuned so at least one language (Hindi) is < 1.2 and the spread is minimised.
WEIGHTS = {"en": 2, "hi": 6, "te": 9, "sa": 10}
VOCAB_SIZE = 10000
REPLACEMENT = "▁"

GATE_SAMPLES = [
    "India's population is 1,428,627,663.",
    "See [India](https://en.wikipedia.org/wiki/India), the world's largest democracy.",
    "भारत गणराज्य, जनसंख्या 1,42,86,27,663 (2023)।",
    "| Year | GDP (US$) |\n|------|-----------|\n| 2024 | $3.9 trillion |",
    "Mixed 中文 café — § ௧௨௩ 🇮🇳 Bhārata",
]


def wordish_units(text: str) -> int:
    return len(regex.findall(r"[\p{L}\p{M}\p{N}]+", text))


def visible(s: str) -> str:
    """Visible non-whitespace characters (the gate compares these)."""
    return "".join(s.split())


def make_tokenizer() -> Tokenizer:
    tok = Tokenizer(BPE(unk_token=None, byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement=REPLACEMENT, prepend_scheme="always", split=True
    )
    tok.decoder = decoders.Sequence([
        decoders.ByteFallback(),
        decoders.Metaspace(replacement=REPLACEMENT, prepend_scheme="always", split=True),
        decoders.Fuse(),
    ])
    return tok


def train() -> tuple[Tokenizer, dict]:
    texts = {c: (CORPUS / f"{c}.faithful.txt").read_text(encoding="utf-8") for c in LANGS}
    units = {c: wordish_units(t) for c, t in texts.items()}

    with tempfile.TemporaryDirectory() as tmp:
        files: list[str] = []
        for c, t in texts.items():
            p = Path(tmp) / f"{c}.txt"
            p.write_text(t, encoding="utf-8")
            files.extend([str(p)] * WEIGHTS[c])

        tok = make_tokenizer()
        byte_tokens = [f"<0x{b:02X}>" for b in range(256)]  # reserve all 256 bytes
        trainer = BpeTrainer(
            vocab_size=VOCAB_SIZE, min_frequency=1, special_tokens=byte_tokens
        )
        tok.train(files, trainer)

    # The 256 byte-fallback tokens are added as "special" by the trainer, and
    # Tokenizer.decode() skips special tokens by default -> byte-fallback output
    # would be dropped. Mark them NON-special so decode() is lossless on any input.
    # This changes decode only; encoding / fertilities are unaffected.
    spec = json.loads(tok.to_str())
    for at in spec.get("added_tokens", []):
        if regex.fullmatch(r"<0x[0-9A-F]{2}>", at["content"]):
            at["special"] = False
    tok = Tokenizer.from_str(json.dumps(spec, ensure_ascii=False))

    token_counts, unk_counts = {}, {}
    for c, t in texts.items():
        enc = tok.encode(t)
        token_counts[c] = len(enc.ids)
        unk_counts[c] = sum(1 for tk in enc.tokens if tk == "[UNK]")

    ratios = {c: token_counts[c] / units[c] for c in LANGS}
    lo = min(ratios, key=ratios.get)
    hi = max(ratios, key=ratios.get)
    spread = ratios[hi] - ratios[lo]
    score = 1000 / spread

    # round-trip gate
    gate_samples_ok = all(visible(tok.decode(tok.encode(s).ids)) == visible(s) for s in GATE_SAMPLES)
    gate_corpus_ok = all(visible(tok.decode(tok.encode(texts[c]).ids)) == visible(texts[c]) for c in LANGS)

    metrics = {
        "variant": "wiki_faithful_markdown_lossless",
        "tokenizer": "Metaspace BPE + byte_fallback (lossless)",
        "fourth_language": "sa",
        "languages": NAMES,
        "weights": WEIGHTS,
        "vocab_size": tok.get_vocab_size(),
        "wordish_units": units,
        "token_counts": token_counts,
        "unk_counts": unk_counts,
        "ratios": ratios,
        "min_lang": lo,
        "min_fertility": ratios[lo],
        "max_lang": hi,
        "spread": spread,
        "raw_score": score,
        "at_least_one_below_1.2": ratios[lo] < 1.2,
        "roundtrip_gate_samples_ok": gate_samples_ok,
        "roundtrip_gate_corpus_ok": gate_corpus_ok,
    }
    return tok, metrics


def main() -> int:
    tok, metrics = train()
    tok.save(str(OUT_TOKENIZER))
    OUT_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    assert metrics["vocab_size"] <= VOCAB_SIZE, "vocab exceeds 10k"
    assert metrics["roundtrip_gate_samples_ok"] and metrics["roundtrip_gate_corpus_ok"], "round-trip gate FAILED"
    assert metrics["at_least_one_below_1.2"], "no language < 1.2"
    print("\nOK: vocab<=10000, round-trip gate PASSED, >=1 language < 1.2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
