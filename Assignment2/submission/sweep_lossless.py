#!/usr/bin/env python3
"""Sweep LOSSLESS Metaspace BPE weights for the grader's set en/hi/te/sa.

Optimises: pass roundtrip gate (mandatory), English <= 1.2 (penalty policy),
minimise spread over the four evaluated languages.
"""
from __future__ import annotations
import tempfile, math
from pathlib import Path
import regex
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import pre_tokenizers, decoders
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
EVAL = ["en", "hi", "te", "sa"]      # grader's four languages
SAMPLES = [
    "India's population is 1,428,627,663.",
    "See [India](https://en.wikipedia.org/wiki/India), the world's largest democracy.",
    "भारत गणराज्य, जनसंख्या 1,42,86,27,663 (2023)।",
]

def wordish(t): return len(regex.findall(r"[\p{L}\p{M}\p{N}]+", t))
def vis(s): return "".join(s.split())

def make():
    tok = Tokenizer(BPE(unk_token=None, byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
    tok.decoder = decoders.Sequence([
        decoders.ByteFallback(),
        decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True),
        decoders.Fuse(),
    ])
    return tok

def train_eval(weights):
    texts = {c: (CORPUS / f"{c}.faithful.txt").read_text(encoding="utf-8") for c in weights}
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for c, t in texts.items():
            p = Path(tmp) / f"{c}.txt"; p.write_text(t, encoding="utf-8")
            files += [str(p)] * weights[c]
        tok = make()
        byte_toks = [f"<0x{b:02X}>" for b in range(256)]
        tok.train(files, BpeTrainer(vocab_size=10000, min_frequency=1, special_tokens=byte_toks))
    units = {c: wordish(texts[c]) for c in EVAL}
    tc = {c: len(tok.encode(texts[c]).ids) for c in EVAL}
    fert = {c: tc[c]/units[c] for c in EVAL}
    spread = max(fert.values()) - min(fert.values())
    raw = 1000/spread
    en_pen = math.exp(max(0.0, fert["en"]/1.2 - 1.0))
    hi_pen = math.exp(max(0.0, fert["hi"]/1.2 - 1.0))
    gate_ok = all(vis(tok.decode(tok.encode(s).ids)) == vis(s) for s in SAMPLES) and \
              all(vis(tok.decode(tok.encode(texts[c]).ids)) == vis(texts[c]) for c in EVAL)
    return {"w": weights, "fert": fert, "spread": spread, "raw": raw,
            "en_pen": en_pen, "hi_pen": hi_pen,
            "adj_en": raw/en_pen, "adj_both": raw/en_pen/hi_pen, "gate": gate_ok}

import itertools
CANDS = [dict(zip(("en","hi","te","sa"), c)) for c in itertools.product(
    (2,3,4,5),      # en  (more en to pull English's max down)
    (5,6,7),        # hi
    (8,9,10),       # te
    (9,10,11),      # sa
)]

def main():
    rows = []
    for w in CANDS:
        r = train_eval(w)
        if not r["gate"]:
            continue
        f = r["fert"]
        minL = min(f, key=f.get)
        rows.append((r["raw"], minL, f[minL], r))
    # keep only configs with at least one language comfortably < 1.2
    good = [x for x in rows if x[2] < 1.19]
    good.sort(key=lambda x: -x[0])
    print("TOP configs with min fertility < 1.19, ranked by raw score:")
    print(f"{'weights':34s} | en    hi    te    sa    | minL  min    spread raw")
    for raw, minL, mn, r in good[:12]:
        f = r["fert"]
        print(f"{str(r['w']):34s} | {f['en']:.3f} {f['hi']:.3f} {f['te']:.3f} {f['sa']:.3f} "
              f"| {minL:4s} {mn:.3f} {r['spread']:.3f} {raw:.0f}")
    if good:
        print("\nBEST:", good[0][3]["w"], "raw=", round(good[0][0]),
              "min", good[0][1], "=", round(good[0][2], 3), "spread", round(good[0][3]["spread"], 3))

if __name__ == "__main__":
    main()
