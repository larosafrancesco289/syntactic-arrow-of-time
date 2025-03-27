import os
import re
import requests
import spacy
from tqdm import tqdm
from collections import defaultdict
import random
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

# Maximum vocabulary size for 16-bit representation
MAX_VOCAB_SIZE = 65535  # 2^16 - 1

# Global variable to store the total vocabulary size
vocab_size = 0


def load_openwebtext_subset(target_size_gb=1):
    """
    Loads a subset of OpenWebText (~10GB) using Hugging Face Datasets.
    Returns the raw text as a single string.
    """
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    total_size_bytes = 0
    target_size_bytes = target_size_gb * 10**9  # Convert GB to bytes
    text_data = []

    print("Loading OpenWebText subset...")
    for example in tqdm(dataset):
        text = example["text"]
        text_size = len(text.encode("utf-8"))
        if total_size_bytes + text_size > target_size_bytes:
            remaining_space = target_size_bytes - total_size_bytes
            text = text[:remaining_space]  # Truncate to fit target size
            text_data.append(text)
            break
        text_data.append(text)
        total_size_bytes += text_size

    print(f"Loaded {total_size_bytes / 10**9:.2f} GB of text")
    return "\n\n".join(text_data)  # Join with double newlines, as in original


def read_data(text):
    """
    Simply returns the text (no file reading needed since we have it in memory).
    """
    return text


def extract_pos_tags(text):
    """
    Extracts words and their part-of-speech tags from the input text using spaCy.

    Processes the entire text through spaCy's NLP pipeline to identify
    tokens and their grammatical properties.

    Args:
        text (str): Raw text to analyze

    Returns:
        list: List of (word, POS tag) tuples for each token in the text
    """
    nlp = spacy.load(
        "en_core_web_sm",
        disable=[
            "parser",
            "ner",
            "attribute_ruler",
            "lemmatizer",
        ],  # Disable unnecessary components
    )  # Note sm is fast and  a cnn model while trf is the transformer model but it's too slow
    nlp.max_length = 10_000_000  # Increase max length for larger texts
    print("Active pipeline components (pipe_names):", nlp.pipe_names)
    chunks = [
        chunk.strip() for chunk in text.split("\n\n") if chunk.strip()
    ]  # Split text into chunks separated by double newlines
    all_tokens = []
    print("Extracting words and part-of-speech tags...")
    # Process each chunk through the NLP pipeline with n_process=-1 to use all available CPU cores
    for doc in tqdm(nlp.pipe(chunks, n_process=-1, batch_size=5000), total=len(chunks)):
        all_tokens.extend([(token.text, token.tag_) for token in doc])
    return all_tokens


def create_tokenizer_dict(all_tokens):
    """
    Creates a dictionary mapping each unique POS tag to a unique integer.

    Args:
        all_tokens (list): List of (word, POS tag) tuples

    Returns:
        dict: Dictionary mapping POS tags to unique integers
    """
    unique_pos_tags = sorted(set(pos_tag for word, pos_tag in all_tokens))
    tokenizer_dict = {pos_tag: idx for idx, pos_tag in enumerate(unique_pos_tags)}
    global vocab_size
    vocab_size = len(unique_pos_tags)
    print("POS tags and their corresponding integers:")
    for pos_tag, token_id in tokenizer_dict.items():
        print(f"{pos_tag}: {token_id}")
    print(f"Vocabulary size: {vocab_size:,}")
    if vocab_size > MAX_VOCAB_SIZE - 1:
        raise ValueError(
            f"Vocabulary size {vocab_size} exceeds maximum allowed size of {MAX_VOCAB_SIZE - 1}"
        )
    return tokenizer_dict


def tokenize_pos_tags(pos_tags, tokenizer_dict):
    """
    Tokenizes each part-of-speech tag into its assigned unique integer.

    Args:
        pos_tags (list): List of part-of-speech tags
        tokenizer_dict (dict): Dictionary mapping POS tags to integers

    Returns:
        list: List of integer tokens representing the input tags
    """
    tokenized = []
    for pos_tag in pos_tags:
        if pos_tag in tokenizer_dict:
            token_id = tokenizer_dict[pos_tag]
            tokenized.append(token_id)
        else:
            print(f"Warning: Unknown POS tag '{pos_tag}'")
            tokenized.append(-1)
    return tokenized


def main():
    """
    Main function to coordinate the entire tokenization pipeline.

    Steps:
    1. Download and read the OpenWebText dataset
    2. Extract part-of-speech tags using spaCy
    3. Create a tokenizer dictionary mapping POS tags to integer ranges
    4. Split data into training and validation sets
    5. Convert POS tags to integer tokens
    6. Save tokenized data and metadata to disk
    """
    # Load ~1GB of OpenWebText
    data = load_openwebtext_subset(target_size_gb=1)

    all_tokens = extract_pos_tags(data)
    tokenizer_dict = create_tokenizer_dict(all_tokens)

    pos_tags = [pos_tag for word, pos_tag in all_tokens]

    # Split into train and validation sets (90/10 split)
    n = len(pos_tags)
    train_pos_tags = pos_tags[: int(n * 0.9)]
    val_pos_tags = pos_tags[int(n * 0.9) :]

    # Tokenize the POS tags into integers
    print("Tokenizing training data...")
    train_tokenized = tokenize_pos_tags(train_pos_tags, tokenizer_dict)
    print("Tokenizing validation data...")
    val_tokenized = tokenize_pos_tags(val_pos_tags, tokenizer_dict)

    # Print token counts
    print(f"Train has {len(train_tokenized):,} tokens")
    print(f"Val has {len(val_tokenized):,} tokens")

    # Print some examples
    print("Example tokens:")
    for i in range(10):
        print(f"Train: {train_tokenized[i]} -> {train_pos_tags[i]}")
        print(f"Val: {val_tokenized[i]} -> {val_pos_tags[i]}")

    # Save to binary files for efficient storage and loading using uint16
    train_tokenized = np.array(train_tokenized, dtype=np.uint16)  # Changed to uint16
    val_tokenized = np.array(val_tokenized, dtype=np.uint16)  # Changed to uint16
    train_tokenized.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
    val_tokenized.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

    # Save metadata with vocabulary size for model training
    meta = {"vocab_size": vocab_size}
    with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    main()
