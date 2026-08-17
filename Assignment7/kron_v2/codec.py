"""
Kronecker V2 codec: the *invertible* byte-position factorization.

V1 (Shravan, 2026) defines the forward codec for a byte string b = (b_1..b_L), L <= d_p:

    kappa(b) = (1/sqrt(L)) * sum_{p=1}^{L}  c_{b_p} (x) p_p          (V1 Eq. 1)

where c_v in R^{d_c} is a one-hot of byte value v (d_c = 256), p_p in R^{d_p} is a
one-hot of position p, and (x) is the Kronecker product. D = d_c * d_p.

KEY OBSERVATION (V2). Reshape kappa(b) into a matrix M in R^{d_c x d_p}:

    M[v, p] = 1/sqrt(L)   if p <= L and b_p == v
            = 0           otherwise

Every position p corresponds to exactly ONE column, and that column is a scaled
one-hot pointing at the byte living at position p. Therefore:

    * length L        = number of non-zero columns (they are the first L columns)
    * byte at pos p   = argmax_v M[v, p]
    * the scale       = column L2 norm = 1/sqrt(L)  (a redundant consistency check)

So kappa is INJECTIVE on strings of length <= d_p and can be inverted exactly by an
argmax down each column. This is a theorem about the encoding, not an empirical
property of a trained network. `decode(encode(s)) == s` for every s (see prove.py).

This module provides:
  * encode_matrix / decode_matrix : the exact invertible pair (the proof object)
  * bytes_of / codec_vector       : precompute-friendly helpers for the model
  * build_codec_table             : the gpu_table variant used as the model input
  * byte_targets                  : per-position class labels used by the V2 output head
"""

import numpy as np

D_C = 256          # byte alphabet (V1 Sec 4.1)
D_P = 32           # max byte positions (V1 Sec 4.2, production default)
EMPTY = 256        # extra output class meaning "no byte at this position"
N_OUT_CLASSES = D_C + 1   # 257: bytes 0..255 plus EMPTY


def token_bytes(s, d_p=D_P):
    """UTF-8 bytes of a token, truncated to the first d_p bytes (UTF-8 safe-ish)."""
    b = s.encode("utf-8")[:d_p]
    return list(b)


# --------------------------------------------------------------------------- #
# The exact invertible pair.  These two functions ARE the mathematical claim.  #
# --------------------------------------------------------------------------- #
def encode_matrix(s, d_c=D_C, d_p=D_P):
    """Forward codec, returned as the (d_c x d_p) matrix view of kappa(b)."""
    b = token_bytes(s, d_p)
    L = len(b)
    M = np.zeros((d_c, d_p), dtype=np.float64)
    if L == 0:
        return M
    scale = 1.0 / np.sqrt(L)
    for p, v in enumerate(b):
        M[v, p] = scale
    return M


def decode_matrix(M):
    """Exact inverse of encode_matrix: recover the byte string from the matrix."""
    # Non-zero columns are the occupied positions (contiguous from the left).
    col_energy = np.abs(M).sum(axis=0)
    occupied = np.nonzero(col_energy > 0)[0]
    if occupied.size == 0:
        return ""
    L = int(occupied.max()) + 1                 # positions are 1..L, contiguous
    bytes_out = [int(np.argmax(M[:, p])) for p in range(L)]
    return bytes(bytes_out).decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Vector form + model-facing tables.                                           #
# --------------------------------------------------------------------------- #
def codec_vector(s, d_c=D_C, d_p=D_P, znorm=True):
    """Flattened D-dim codec vector, with V1 Sec 3.3 per-token z-normalization."""
    M = encode_matrix(s, d_c, d_p)
    v = M.reshape(-1)                            # row-major: index = byte*d_p + pos
    if znorm:
        mu, sd = v.mean(), v.std()
        v = (v - mu) / (sd + 1e-8)
    return v.astype(np.float32)


def build_codec_table(vocab, d_c=D_C, d_p=D_P, znorm=True):
    """[|V|, D] float32 table (gpu_table variant): row i = codec of vocab[i]."""
    D = d_c * d_p
    K = np.zeros((len(vocab), D), dtype=np.float32)
    for i, tok in enumerate(vocab):
        K[i] = codec_vector(tok, d_c, d_p, znorm)
    return K


def byte_targets(vocab, d_p=D_P):
    """[|V|, d_p] int64 class labels for the V2 head: byte value per position,
    or EMPTY(256) where the token has no byte.  argmax over these labels == token."""
    T = np.full((len(vocab), d_p), EMPTY, dtype=np.int64)
    for i, tok in enumerate(vocab):
        b = token_bytes(tok, d_p)
        for p, v in enumerate(b):
            T[i, p] = v
    return T


def labels_to_string(labels):
    """Decode a length-d_p label vector (from the V2 head) back to a string.
    Stops at the first EMPTY; the byte-composition is predicted in one shot."""
    out = []
    for v in labels:
        if v == EMPTY:
            break
        out.append(int(v))
    return bytes(out).decode("utf-8", errors="replace")
