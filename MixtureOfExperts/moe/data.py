"""Byte-level text data. We tokenize raw UTF-8 bytes (vocab = 256), so there is no
vocab file and the token embedding is tiny — which keeps the model's parameters
concentrated in the feed-forward / MoE block we are actually studying.

Source text is a slice of TinyStories (simple, coherent English), streamed from the
HuggingFace hub and cached as uint8 .bin files.
"""
from __future__ import annotations

import os

import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def prepare(train_bytes: int = 40_000_000, val_bytes: int = 2_000_000):
    os.makedirs(DATA_DIR, exist_ok=True)
    train_bin = os.path.join(DATA_DIR, "train.bin")
    val_bin = os.path.join(DATA_DIR, "val.bin")

    def big_enough(p, n):
        return os.path.exists(p) and os.path.getsize(p) >= n

    if big_enough(train_bin, train_bytes) and big_enough(val_bin, val_bytes):
        print(f"[data] cached: {train_bin}, {val_bin}")
        return train_bin, val_bin

    from datasets import load_dataset

    def stream_to_bin(split, target, path):
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        buf = bytearray()
        for ex in ds:
            buf.extend(ex["text"].encode("utf-8"))
            buf.append(0)  # separator byte between stories
            if len(buf) >= target:
                break
        arr = np.frombuffer(bytes(buf[:target]), dtype=np.uint8)
        arr.tofile(path)
        print(f"[data] wrote {len(arr):,} bytes -> {path}")

    print("[data] streaming TinyStories as raw bytes…")
    stream_to_bin("train", train_bytes, train_bin)
    stream_to_bin("validation", val_bytes, val_bin)
    return train_bin, val_bin


class ByteData:
    def __init__(self, bin_path: str):
        self.data = np.memmap(bin_path, dtype=np.uint8, mode="r")

    def __len__(self):
        return len(self.data)

    def get_batch(self, batch_size, block_size, device):
        ix = torch.randint(len(self.data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)
