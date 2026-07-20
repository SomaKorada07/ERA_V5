#!/usr/bin/env python3
"""Prototype LOSSLESS tokenizers and measure the roundtrip gate + fertility.

Gate: for every text, strip whitespace from input and from decode(encode(text));
the remaining visible characters must be identical.
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
LANGS = ["en", "hi", "te", "la"]
SAMPLES = [
    "India's population is 1,428,627,663.",
    "See [India](https://en.wikipedia.org/wiki/India) — the world's largest democracy.",
    "भारत गणराज्य, जनसंख्या 1,42,86,27,663 (2023)।",
    "| Year | GDP (US$) |\n|------|-----------|\n| 2024 | $3.9 trillion |",
]

def wordish(text): return len(regex.findall(r"[\p{L}\p{M}\p{N}]+", text))
def vis(s): return "".join(s.split())

def texts():
    return {c: (CORPUS / f"{c}.faithful.txt").read_text(encoding="utf-8") for c in LANGS}

def train_bytelevel(weights):
    ts = texts()
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for c, t in ts.items():
            p = Path(tmp) / f"{c}.txt"; p.write_text(t, encoding="utf-8")
            files += [str(p)] * weights[c]
        tok = Tokenizer(BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tok.decoder = decoders.ByteLevel()
        tr = BpeTrainer(vocab_size=10000, min_frequency=1,
                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[])
        tok.train(files, tr)
    return tok

def gate(tok):
    ts = texts()
    bad = 0
    for s in SAMPLES:
        if vis(tok.decode(tok.encode(s).ids)) != vis(s): bad += 1
    # full corpus roundtrip
    corpus_ok = True
    for c, t in ts.items():
        if vis(tok.decode(tok.encode(t).ids)) != vis(t): corpus_ok = False
    return bad, corpus_ok

def report(name, tok):
    ts = texts()
    units = {c: wordish(t) for c, t in ts.items()}
    tc = {c: len(tok.encode(t).ids) for c, t in ts.items()}
    fert = {c: tc[c]/units[c] for c in LANGS}
    spread = max(fert.values()) - min(fert.values())
    bad, corpus_ok = gate(tok)
    en_pen = math.exp(max(0.0, fert["en"]/1.2 - 1.0))
    print(f"\n== {name} == vocab={tok.get_vocab_size()}")
    for c in LANGS: print(f"   {c}: fert={fert[c]:.4f} tokens={tc[c]}")
    print(f"   spread={spread:.4f} raw={1000/spread:.0f} | EN={fert['en']:.4f} pen=x{en_pen:.3f} adj={1000/spread/en_pen:.0f}")
    print(f"   GATE: sample_fail={bad}/{len(SAMPLES)} corpus_roundtrip_ok={corpus_ok}")
    # show one sample roundtrip
    s = SAMPLES[0]
    print(f"   sample: {tok.decode(tok.encode(s).ids)!r}")
    return fert

def train_metaspace(weights, prepend="always"):
    ts = texts()
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for c, t in ts.items():
            p = Path(tmp) / f"{c}.txt"; p.write_text(t, encoding="utf-8")
            files += [str(p)] * weights[c]
        tok = Tokenizer(BPE(unk_token=None, byte_fallback=True))
        tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=prepend, split=True)
        tok.decoder = decoders.Sequence([
            decoders.ByteFallback(),
            decoders.Metaspace(replacement="▁", prepend_scheme=prepend, split=True),
            decoders.Fuse(),
        ])
        # reserve the 256 byte tokens so byte_fallback is fully covered
        byte_toks = [f"<0x{b:02X}>" for b in range(256)]
        tr = BpeTrainer(vocab_size=10000, min_frequency=1, special_tokens=byte_toks)
        tok.train(files, tr)
    return tok

for w in [{"en":3,"hi":8,"te":11,"la":7}, {"en":4,"hi":6,"te":8,"la":5}, {"en":5,"hi":5,"te":7,"la":4}]:
    tok = train_bytelevel(w)
    report(f"ByteLevel {w}", tok)

for w in [{"en":3,"hi":8,"te":11,"la":7}, {"en":4,"hi":6,"te":8,"la":5}, {"en":3,"hi":6,"te":8,"la":5}]:
    tok = train_metaspace(w)
    report(f"Metaspace {w}", tok)
