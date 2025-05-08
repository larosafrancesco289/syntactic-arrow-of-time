#!/usr/bin/env python
"""
Tokenise a size-bounded slice of a local CC-100 .xz shard with the
GPT-2 tokenizer and write train.bin / val.bin plus meta.pkl.

>  • Supports streaming decompression so only the requested GB are read.
>  • Uses uint16 because GPT-2's vocab (50 257) fits comfortably.
"""
from __future__ import annotations
import lzma
import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import GPT2TokenizerFast

# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────
CC100_XZ_PATH = Path("/home/larosa/en.txt.xz")  # ← point at your file
TARGET_SIZE_GB = 2.0  # uncompressed text size to read
VAL_FRACTION = 0.05  # 5 % for validation
OUT_DIR = Path(__file__).parent
DTYPE = np.uint16  # GPT-2 vocab fits in 16 bits
# ────────────────────────────────────────────────────────────────


def load_cc100_subset(fp: str | Path, target_size_gb: float) -> list[str]:
    """
    Stream-decompress `fp` until `target_size_gb` UTF-8 bytes are collected.
    Returns a list of documents (one per line, original newlines stripped).
    """
    target_bytes = int(target_size_gb * 1_000_000_000)
    docs, total = [], 0

    pbar = tqdm(
        total=target_bytes,
        desc="Reading CC-100",
        unit="B",
        unit_scale=True,
        unit_divisor=1000,
        dynamic_ncols=True,
    )
    with lzma.open(fp, mode="rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if total >= target_bytes:
                break
            line_b = line.encode("utf-8")
            b = len(line_b)
            if total + b > target_bytes:
                line_b = line_b[: target_bytes - total]
                line = line_b.decode("utf-8", errors="ignore")
            docs.append(line.rstrip("\n"))
            total += len(line_b)
            pbar.update(len(line_b))
            if total >= target_bytes:
                break
    pbar.close()

    print(f"Loaded {total/1_000_000_000:.2f} GB from {fp}")
    return docs


def tokenize_docs(docs: list[str], tokenizer: GPT2TokenizerFast) -> list[int]:
    """
    Encode each doc and append GPT-2’s EOS id (50256) between documents,
    returning one flat list of token IDs.
    """
    eos = tokenizer.eos_token_id
    ids = []
    for doc in tqdm(docs, desc="Tokenising"):
        ids.extend(tokenizer.encode(doc, add_special_tokens=False) + [eos])
    return ids


def main() -> None:
    # ─────────── Load data ───────────
    docs = load_cc100_subset(CC100_XZ_PATH, TARGET_SIZE_GB)

    # ────────── Tokenise ─────────────
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ids = tokenize_docs(docs, tok)

    # ───────── Train / Val split ─────
    split = int(len(ids) * (1 - VAL_FRACTION))
    train_ids = np.asarray(ids[:split], dtype=DTYPE)
    val_ids = np.asarray(ids[split:], dtype=DTYPE)

    print(
        f"Total tokens {len(ids):,}  → train {len(train_ids):,} | "
        f"val {len(val_ids):,}"
    )

    # ───────── Persist ───────────────
    (OUT_DIR / "train.bin").write_bytes(train_ids.tobytes())
    (OUT_DIR / "val.bin").write_bytes(val_ids.tobytes())

    meta = dict(
        vocab_size=tok.vocab_size,  # 50 257
        eos_token_id=tok.eos_token_id,  # 50256
        dtype=str(DTYPE),
        train_tokens=len(train_ids),
        val_tokens=len(val_ids),
        source="cc100_en",
        raw_gb=TARGET_SIZE_GB,
    )
    with open(OUT_DIR / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print("Done! Files saved to", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
