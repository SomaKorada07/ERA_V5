#!/usr/bin/env python3
"""
Evaluate tokenizer.json on the faithful Wikipedia Markdown corpus for the
grader's four languages: English, Hindi, Telugu, Sanskrit.

Checks BOTH:
  1. the round-trip GATE  -- decode(encode(text)) must preserve the same visible
     (non-whitespace) characters, for every corpus and for adversarial samples;
  2. fertility / spread / score, with the "at least one language < 1.2" rule.

    wordish_units(text) = number of [\\p{L}\\p{M}\\p{N}]+ runs
    fertility(lang)     = tokens(lang) / wordish_units(lang)
    spread              = max(fertility) - min(fertility)
    raw_score           = 1000 / spread

Run:
    pip install tokenizers regex
    python evaluate_tokenizer.py
"""
from __future__ import annotations

import json
from pathlib import Path

import regex
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
TOKENIZER = ROOT / "tokenizer.json"
LANGS = ["en", "hi", "te", "sa"]
NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "sa": "Sanskrit"}

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
    return "".join(s.split())


def main() -> int:
    tok = Tokenizer.from_file(str(TOKENIZER))

    # ---- round-trip gate (decode uses the default skip_special_tokens=True) ----
    texts = {c: (CORPUS / f"{c}.faithful.txt").read_text(encoding="utf-8") for c in LANGS}
    print("ROUND-TRIP GATE  (decode(encode(x)) preserves visible non-whitespace chars)")
    gate_ok = True
    for s in GATE_SAMPLES:
        ok = visible(tok.decode(tok.encode(s).ids)) == visible(s)
        gate_ok = gate_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] sample: {s[:46]!r}")
    for c in LANGS:
        ok = visible(tok.decode(tok.encode(texts[c]).ids)) == visible(texts[c])
        gate_ok = gate_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] full {NAMES[c]} corpus")
    print(f"  => GATE {'PASSED' if gate_ok else 'FAILED'}\n")

    # ---- fertility / score ----
    rows = {}
    for c in LANGS:
        enc = tok.encode(texts[c])
        rows[c] = {
            "name": NAMES[c],
            "tokens": len(enc.ids),
            "wordish_units": wordish_units(texts[c]),
            "ratio": len(enc.ids) / wordish_units(texts[c]),
        }
    ratios = {c: rows[c]["ratio"] for c in LANGS}
    lo = min(ratios, key=ratios.get)
    hi = max(ratios, key=ratios.get)
    spread = ratios[hi] - ratios[lo]
    score = 1000 / spread

    print(f"{'language':10s}{'tokens':>9s}{'word-ish':>10s}{'fertility':>11s}")
    for c in sorted(LANGS, key=lambda x: ratios[x]):
        r = rows[c]
        print(f"{r['name']:10s}{r['tokens']:>9d}{r['wordish_units']:>10d}{r['ratio']:>11.4f}")
    print()
    print(f"min fertility : {ratios[lo]:.4f} [{NAMES[lo]}]   (rule: at least one language < 1.2 -> {ratios[lo] < 1.2})")
    print(f"max fertility : {ratios[hi]:.4f} [{NAMES[hi]}]")
    print(f"spread        : {spread:.4f}")
    print(f"raw score     : {score:.1f}   = 1000 / spread")
    print(f"vocab size    : {tok.get_vocab_size()}")

    result = {
        "roundtrip_gate_passed": gate_ok,
        "rows": rows,
        "min_lang": NAMES[lo],
        "min_fertility": ratios[lo],
        "at_least_one_below_1.2": ratios[lo] < 1.2,
        "max_lang": NAMES[hi],
        "spread": spread,
        "raw_score": score,
        "vocab_size": tok.get_vocab_size(),
    }
    print("\n" + json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
