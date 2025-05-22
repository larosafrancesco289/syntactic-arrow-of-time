"""
Evaluate average token-level cross-entropy *per sentence* on a 1 MB slice of CC-100
for two already-trained GPT models (GPT-1-sized and GPT-nano-sized).

Usage
-----
python evaluate_sentence_loss.py \
    --cc100_shard /home/larosa/en.txt.xz \
    --checkpoint_gpt1 out/cc100_gpt1/ckpt.pt \
    --checkpoint_nano out/cc100_nano/ckpt.pt \
    --meta_path data/cc100/meta.pkl  # tokenizer dict created during training

The script will:
1. Stream **exactly 1 MB** (≈1 000 000 bytes) of UTF-8 text from the given CC-100 shard
   using `tokenizer.load_cc100_subset`.
2. Split that text into *sentences* with spaCy (keeping punctuation).
3. Re-tokenise every sentence with the *same* POS-tokeniser that was used in
   training (requires `meta.pkl`).
4. Feed each sentence independently through the model and collect the mean
   cross-entropy (equivalent to *validation loss*).
5. Report the average cross-entropy over all sentences and the corresponding
   perplexity.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import spacy
except ImportError:
    sys.exit(
        "spaCy is required: pip install spacy && python -m spacy download en_core_web_sm"
    )

# Project-specific utilities
from data.cc100.tokenizer import load_cc100_subset, tokenize_pos_tags
from model import GPT, GPTConfig

# ────────────────────────────────────────────────────────────────────────────────
# User configuration                                                            │
# ────────────────────────────────────────────────────────────────────────────────
# Fill in **absolute** paths that point at your local files.
# Leave DEVICE to the default unless you want to force CPU.
# ------------------------------------------------------------------------------
CC100_SHARD = Path("/raid/en.txt.xz")
CHECKPOINT_GPT1 = Path("out/cc100_gpt1/ckpt.pt")
CHECKPOINT_NANO = Path("out/cc100_nano/ckpt.pt")
META_PATH = Path("data/cc100/meta.pkl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

######################################################################
# Helpers
######################################################################


def load_tokeniser(meta_path: Path):
    """Load the POS-tokeniser dictionary & vocab size that were stored at training time."""
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta["tokenizer_dict"], int(meta["vocab_size"])


def encode_sentence(sentence: str, nlp, tokenizer_dict, bos_id):
    doc = nlp(sentence)
    pos_tags = [t.tag_ for t in doc]
    ids = tokenize_pos_tags(pos_tags, tokenizer_dict)
    return [bos_id] + ids


def load_model(ckpt_path: Path, device: torch.device):
    """Load a model checkpoint that was saved by `train.py`."""
    ckpt = torch.load(ckpt_path, map_location=device)
    config = GPTConfig(**ckpt["model_args"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def sentence_loss(model: GPT, token_ids, device):
    if len(token_ids) < 2:
        return None  # cannot compute next-token loss on <2 tokens
    # Truncate to model context window.
    token_ids = token_ids[: model.config.block_size]
    x = torch.tensor(token_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(token_ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        _, loss = model(x, y)
    return loss.item()


def evaluate(model: GPT, sentences, nlp, tokenizer_dict, device):
    losses = []
    for sent in tqdm(sentences, desc="eval", ncols=80):
        ids = encode_sentence(sent, nlp, tokenizer_dict, BOS_ID)
        l = sentence_loss(model, ids, device)
        if l is not None:
            losses.append(l)
    return sum(losses) / len(losses)


def evaluate_full_context(model: GPT, token_stream, device):
    """Cross-entropy over one contiguous stream (no resets)."""
    losses, n_tokens = 0.0, 0
    block = model.config.block_size
    # iterate over overlapping target pairs
    for i in range(0, len(token_stream) - block - 1, block):
        x = torch.tensor(
            token_stream[i : i + block], dtype=torch.long, device=device
        ).unsqueeze(0)
        y = torch.tensor(
            token_stream[i + 1 : i + block + 1], dtype=torch.long, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            _, loss = model(x, y)
        losses += loss.item() * y.numel()
        n_tokens += y.numel()
    return losses / n_tokens


######################################################################
# Main
######################################################################


def parse_args():
    """Return argparse.Namespace overriding any globals supplied on the CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cc100_shard", type=Path)
    parser.add_argument("--checkpoint_gpt1", type=Path)
    parser.add_argument("--checkpoint_nano", type=Path)
    parser.add_argument("--meta_path", type=Path)
    parser.add_argument("--device")
    args, _ = parser.parse_known_args()
    return args


def main():
    # 0. Collect parameters -----------------------------------------------------
    args = parse_args()
    cc100_shard = args.cc100_shard or CC100_SHARD
    checkpoint_gpt1 = args.checkpoint_gpt1 or CHECKPOINT_GPT1
    checkpoint_nano = args.checkpoint_nano or CHECKPOINT_NANO
    meta_path = args.meta_path or META_PATH
    device = torch.device(args.device or DEVICE)

    # ------------------------------------------------------------------
    # 1. Data: 1 MB of CC-100, sentence-split
    # ------------------------------------------------------------------
    print("\n⇢ Loading 1 MB of CC-100 text ...")
    text = load_cc100_subset(cc100_shard, target_size_gb=0.001)

    print("⇢ Splitting into sentences ...")
    nlp_sent = spacy.blank("en")
    # ensure we have a cheap sentenciser in the pipeline
    if "sentencizer" not in nlp_sent.pipe_names:
        nlp_sent.add_pipe("sentencizer")
    nlp_sent.max_length = 10_000_000
    doc = nlp_sent(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    print(f"   {len(sentences):,} sentences")

    # ------------------------------------------------------------------
    # 2. Tokeniser dictionary
    # ------------------------------------------------------------------
    tokenizer_dict, vocab_size = load_tokeniser(meta_path)
    global BOS_ID
    BOS_ID = vocab_size  # BOS token is the last in the vocab

    # Share the same spaCy object for POS-tagging (tagger only)
    nlp_tag = spacy.load(
        "en_core_web_sm",
        disable=[
            "parser",
            "lemmatizer",
            "ner",
            "textcat",
            "attribute_ruler",
        ],
    )

    # ------------------------------------------------------------------
    # 3. Models
    # ------------------------------------------------------------------
    print("\n⇢ Loading GPT-1 checkpoint …")
    model_gpt1 = load_model(checkpoint_gpt1, device)

    print("⇢ Loading GPT-nano checkpoint …")
    model_nano = load_model(checkpoint_nano, device)

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------
    # build one big stream (still includes the BOS before each sentence)
    token_stream = [
        t
        for ids in (
            encode_sentence(s, nlp_tag, tokenizer_dict, BOS_ID) for s in sentences
        )
        for t in ids
    ]

    print("\n⇢ Evaluating GPT-1 …")
    loss_sent_gpt1 = evaluate(model_gpt1, sentences, nlp_tag, tokenizer_dict, device)
    loss_full_gpt1 = evaluate_full_context(model_gpt1, token_stream, device)

    print("⇢ Evaluating GPT-nano …")
    loss_sent_nano = evaluate(model_nano, sentences, nlp_tag, tokenizer_dict, device)
    loss_full_nano = evaluate_full_context(model_nano, token_stream, device)

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------
    print("\n================ RESULTS =================")
    print("          |  sentence-avg |  full-context ")
    print(f"GPT-1     | {loss_sent_gpt1:6.4f}      | {loss_full_gpt1:6.4f}")
    print(f"GPT-nano  | {loss_sent_nano:6.4f}      | {loss_full_nano:6.4f}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
