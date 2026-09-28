"""TinyStories data: stream, tokenize with GPT-2 BPE, cache as uint16 token bins."""
from __future__ import annotations

import os

import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _encoder():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def prepare(target_train_tokens: int = 60_000_000, target_val_tokens: int = 2_000_000):
    """Tokenize a slice of TinyStories into data/{train,val}.bin (uint16).

    We take slightly more than the 50M-token training budget so the run sees fresh
    text rather than cycling. Idempotent: skips work if the bins already exist and
    are large enough.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    train_bin = os.path.join(DATA_DIR, "train.bin")
    val_bin = os.path.join(DATA_DIR, "val.bin")

    def _big_enough(path, n):
        return os.path.exists(path) and os.path.getsize(path) >= 2 * n

    if _big_enough(train_bin, target_train_tokens) and _big_enough(val_bin, target_val_tokens):
        print(f"[data] cached: {train_bin}, {val_bin}")
        return train_bin, val_bin

    from datasets import load_dataset
    enc = _encoder()
    eot = enc.eot_token

    def stream_to_bin(split, target, path):
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        buf = np.empty(target + 4096, dtype=np.uint16)
        n = 0
        for ex in ds:
            ids = enc.encode_ordinary(ex["text"])
            ids.append(eot)
            k = len(ids)
            if n + k > len(buf):
                buf = np.concatenate([buf, np.empty(target, dtype=np.uint16)])
            buf[n:n + k] = np.array(ids, dtype=np.uint16)
            n += k
            if n >= target:
                break
        buf[:n].tofile(path)
        print(f"[data] wrote {n:,} tokens -> {path}")
        return n

    print("[data] tokenizing TinyStories (streaming)…")
    stream_to_bin("train", target_train_tokens, train_bin)
    stream_to_bin("validation", target_val_tokens, val_bin)
    return train_bin, val_bin


class TokenData:
    def __init__(self, bin_path: str):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.data)

    def get_batch(self, batch_size: int, block_size: int, device: str):
        ix = torch.randint(len(self.data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        if device.startswith("mps") or device.startswith("cuda"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
