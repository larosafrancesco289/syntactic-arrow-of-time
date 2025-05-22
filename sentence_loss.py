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


######################################################################
# Helpers
######################################################################


def load_tokeniser(meta_path: Path):
    """Load the POS-tokeniser dictionary & vocab size that were stored at training time."""
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta["tokenizer_dict"], int(meta["vocab_size"])


def encode_sentence(sentence: str, nlp, tokenizer_dict):
    """Return a list of integer token IDs for *one* sentence."""
    doc = nlp(sentence)
    pos_tags = [t.tag_ for t in doc]
    return tokenize_pos_tags(pos_tags, tokenizer_dict)


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
        ids = encode_sentence(sent, nlp, tokenizer_dict)
        l = sentence_loss(model, ids, device)
        if l is not None:
            losses.append(l)
    return sum(losses) / len(losses)


######################################################################
# Main
######################################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cc100_shard",
        type=Path,
        required=True,
        help="Path to the local .xz shard of CC-100",
    )
    parser.add_argument("--checkpoint_gpt1", type=Path, required=True)
    parser.add_argument("--checkpoint_nano", type=Path, required=True)
    parser.add_argument(
        "--meta_path",
        type=Path,
        required=True,
        help="Path to meta.pkl that matches the training tokenizer",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # 1. Data: 1 MB of CC-100, sentence-split
    # ------------------------------------------------------------------
    print("\n⇢ Loading 1 MB of CC-100 text ...")
    text = load_cc100_subset(args.cc100_shard, target_size_gb=0.001)

    print("⇢ Splitting into sentences ...")
    nlp_sent = spacy.load("en_core_web_sm")
    # ensure we have a cheap sentenciser in the pipeline
    if "sentencizer" not in nlp_sent.pipe_names:
        nlp_sent.add_pipe("sentencizer")
    doc = nlp_sent(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    print(f"   {len(sentences):,} sentences")

    # ------------------------------------------------------------------
    # 2. Tokeniser dictionary
    # ------------------------------------------------------------------
    tokenizer_dict, vocab_size = load_tokeniser(args.meta_path)

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
    model_gpt1 = load_model(args.checkpoint_gpt1, device)

    print("⇢ Loading GPT-nano checkpoint …")
    model_nano = load_model(args.checkpoint_nano, device)

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------
    print("\n⇢ Evaluating GPT-1 …")
    loss_gpt1 = evaluate(model_gpt1, sentences, nlp_tag, tokenizer_dict, device)
    ppl_gpt1 = torch.exp(torch.tensor(loss_gpt1)).item()

    print("⇢ Evaluating GPT-nano …")
    loss_nano = evaluate(model_nano, sentences, nlp_tag, tokenizer_dict, device)
    ppl_nano = torch.exp(torch.tensor(loss_nano)).item()

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------
    print("\n================ RESULTS ================")
    print(f"GPT-1    | cross-entropy {loss_gpt1:6.4f} | perplexity {ppl_gpt1:6.2f}")
    print(f"GPT-nano | cross-entropy {loss_nano:6.4f} | perplexity {ppl_nano:6.2f}")
    print("========================================\n")


if __name__ == "__main__":
    main()
