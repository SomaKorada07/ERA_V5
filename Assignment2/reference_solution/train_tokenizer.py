#!/usr/bin/env python3
"""
Train the shared 10k BPE tokenizer for the faithful Markdown corpus.

Run:
    python build_wiki_faithful_markdown.py
    python train_tokenizer.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import regex
from tokenizers import Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Replace, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
OUT_TOKENIZER = ROOT / "tokenizer.json"
OUT_METRICS = ROOT / "metrics.json"

LANGS = ["en", "hi", "te", "mai"]
WEIGHTS = {"en": 3, "hi": 4, "te": 4, "mai": 2}


def wordish_units(text: str) -> int:
    return len(regex.findall(r"[\p{L}\p{M}\p{N}]+", text))


def make_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence(
        [
            NFKC(),
            Replace(Regex(r"[^\p{L}\p{M}\p{N}]+"), " "),
        ]
    )
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def train() -> tuple[Tokenizer, dict]:
    texts = {
        code: (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        for code in LANGS
    }
    units = {code: wordish_units(text) for code, text in texts.items()}

    with tempfile.TemporaryDirectory() as tmp:
        files: list[str] = []
        tmpdir = Path(tmp)
        for code, text in texts.items():
            path = tmpdir / f"{code}.txt"
            path.write_text(text, encoding="utf-8")
            files.extend([str(path)] * WEIGHTS[code])

        tokenizer = make_tokenizer()
        trainer = BpeTrainer(
            vocab_size=10000,
            min_frequency=1,
            special_tokens=["[UNK]"],
        )
        tokenizer.train(files, trainer)

    token_counts = {code: len(tokenizer.encode(text).ids) for code, text in texts.items()}
    ratios = {code: token_counts[code] / units[code] for code in LANGS}
    spread = max(ratios.values()) - min(ratios.values())
    score = 1000 / spread

    metrics = {
        "variant": "wiki_faithful_markdown",
        "languages": {
            "en": "English",
            "hi": "Hindi",
            "te": "Telugu",
            "mai": "Maithili",
        },
        "weights": WEIGHTS,
        "vocab_size": tokenizer.get_vocab_size(),
        "wordish_units": units,
        "token_counts": token_counts,
        "ratios": ratios,
        "spread": spread,
        "score": score,
    }
    return tokenizer, metrics


def main() -> int:
    tokenizer, metrics = train()
    tokenizer.save(str(OUT_TOKENIZER))
    OUT_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
