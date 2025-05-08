#!/usr/bin/env python
"""
Tokenise an OpenWebText subset with the standard GPT-2 tokenizer
and save train/val .bin files plus a meta.pkl describing the setup.
"""
import os
import pickle
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2TokenizerFast

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
TARGET_SIZE_GB = 2  # change as needed (≈ 2 GB of UTF-8 text)
VAL_FRACTION = 0.05  # 5 % for validation
OUT_DIR = Path(__file__).parent
DTYPE = np.uint16  # GPT-2 vocab (50 257) fits in uint16
# ------------------------------------------------------------


def load_openwebtext_subset(target_size_gb: int = 1) -> list[str]:
    """
    Stream OpenWebText until `target_size_gb` of raw UTF-8 bytes is reached.
    Returns a list of documents (strings).
    """
    stream = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    target_bytes = target_size_gb * 10**9
    docs, total = [], 0
    print(f"Streaming OpenWebText until we hit ≈{target_bytes/10**6:.0f} MB …")
    for ex in tqdm(stream):
        txt = ex["text"]
        size = len(txt.encode("utf-8"))
        if total + size > target_bytes:
            # truncate last doc so we don’t overshoot
            remain = target_bytes - total
            txt = txt[:remain]
            docs.append(txt)
            break
        docs.append(txt)
        total += size
    print(f"Loaded {total/10**9:.2f} GB across {len(docs):,} documents")
    return docs


def tokenize_docs(docs: list[str], tokenizer: GPT2TokenizerFast) -> list[int]:
    """
    Encode each doc, append an ⟨|endoftext|⟩ between docs,
    and return a single flat list of token ids.
    """
    eot_id = tokenizer.eos_token_id  # 50256 for plain gpt2
    all_ids = []
    for doc in tqdm(docs, desc="Tokenising"):
        ids = tokenizer.encode(doc, add_special_tokens=False)
        all_ids.extend(ids + [eot_id])
    return all_ids


def main():
    # ---------------- Load data ----------------
    docs = load_openwebtext_subset(TARGET_SIZE_GB)

    # ---------------- Tokenise -----------------
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    token_ids = tokenize_docs(docs, tokenizer)

    # --------------- Train / Val ---------------
    n = len(token_ids)
    split = int(n * (1 - VAL_FRACTION))
    train_ids = np.array(token_ids[:split], dtype=DTYPE)
    val_ids = np.array(token_ids[split:], dtype=DTYPE)

    print(
        f"Total tokens: {n:,}  →  train {len(train_ids):,} | " f"val {len(val_ids):,}"
    )

    # ----------- Persist to disk ---------------
    train_ids.tofile(OUT_DIR / "train.bin")
    val_ids.tofile(OUT_DIR / "val.bin")

    meta = {
        "vocab_size": tokenizer.vocab_size,  # 50 257
        "eos_token_id": tokenizer.eos_token_id,  # 50256
        "dtype": str(DTYPE),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
    }
    with open(OUT_DIR / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print("Done! Files written to", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
