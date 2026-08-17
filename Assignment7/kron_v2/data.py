"""
A small templated-grammar English corpus.

The grammar has strong local structure so next-token prediction is genuinely
learnable at tiny scale, giving a clean, legible comparison between the two output
heads. Words vary in byte-length (2..12) so the codec pathway is exercised.
"""

import random

DET   = ["the", "a", "another", "each", "every"]
ADJ   = ["quick", "silent", "ancient", "curious", "golden", "restless",
         "gentle", "clever", "hollow", "brilliant", "weary", "distant"]
NOUN  = ["fox", "scholar", "river", "engine", "sparrow", "mountain",
         "kingdom", "machine", "wanderer", "lantern", "harbor", "signal"]
VERB  = ["watches", "follows", "remembers", "builds", "crosses", "questions",
         "guards", "measures", "abandons", "discovers", "repairs", "carries"]
PREP  = ["beyond", "beneath", "within", "beside", "against", "toward"]
CONJ  = ["and", "but", "while", "because", "although"]

TEMPLATES = [
    "{D} {A} {N} {V} {D} {A} {N} .",
    "{D} {N} {V} {D} {N} {P} {D} {A} {N} .",
    "{C} {D} {A} {N} {V} {D} {N} .",
    "{D} {A} {N} {V} {D} {N} {C} {D} {N} {V} {D} {A} {N} .",
    "{P} {D} {N} {D} {A} {N} {V} {D} {N} .",
]


def _fill(t, rng):
    out = []
    i = 0
    while i < len(t):
        if t[i] == "{":
            j = t.index("}", i)
            key = t[i + 1:j]
            out.append({"D": DET, "A": ADJ, "N": NOUN,
                        "V": VERB, "P": PREP, "C": CONJ}[key][
                        rng.randrange(len({"D": DET, "A": ADJ, "N": NOUN,
                                           "V": VERB, "P": PREP, "C": CONJ}[key]))])
            i = j + 1
        else:
            i += 1
    return out


def generate(n_sentences=4000, seed=0):
    rng = random.Random(seed)
    tokens = []
    for _ in range(n_sentences):
        t = TEMPLATES[rng.randrange(len(TEMPLATES))]
        tokens.extend(_fill(t, rng))
    return tokens


def build_vocab(tokens):
    vocab = sorted(set(tokens))
    stoi = {t: i for i, t in enumerate(vocab)}
    return vocab, stoi


if __name__ == "__main__":
    toks = generate(20, seed=1)
    print(" ".join(toks[:60]))
    v, _ = build_vocab(toks)
    print("vocab size (sample)", len(v))
