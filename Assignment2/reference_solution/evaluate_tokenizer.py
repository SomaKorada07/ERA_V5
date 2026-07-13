#!/usr/bin/env python3
"""Evaluate tokenizer.json on the faithful Markdown corpus."""
from __future__ import annotations

import json
import math
from pathlib import Path

import regex
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
TOKENIZER = ROOT / "tokenizer.json"
LANGS = ["en", "hi", "te", "mai"]


def wordish_units(text: str) -> int:
    return len(regex.findall(r"[\p{L}\p{M}\p{N}]+", text))


def main() -> int:
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    rows = {}
    for code in LANGS:
        text = (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        units = wordish_units(text)
        tokens = len(tokenizer.encode(text).ids)
        rows[code] = {"tokens": tokens, "wordish_units": units, "ratio": tokens / units}

    ratios = [row["ratio"] for row in rows.values()]
    spread = max(ratios) - min(ratios)
    score = 1000 / spread
    hindi_penalty = math.exp(max(0.0, rows["hi"]["ratio"] / 1.2 - 1.0))
    result = {
        "rows": rows,
        "spread": spread,
        "score": score,
        "hindi_exp1_penalty_factor": hindi_penalty,
        "hindi_exp1_adjusted_score": score / hindi_penalty,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
