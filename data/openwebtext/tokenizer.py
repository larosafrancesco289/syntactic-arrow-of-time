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
    for doc in tqdm(nlp.pipe(chunks, n_process=20, batch_size=5000), total=len(chunks)):
        all_tokens.extend([(token.text, token.tag_) for token in doc])
    return all_tokens


def create_tokenizer_dict(all_tokens):
    """
    Creates a dictionary mapping POS tags to unique integer ranges.

    Assigns a continuous range of integers to each POS tag based on the number of
    unique words, with 'NNP' capped at 2000 tokens before scaling others.

    Args:
        all_tokens (list): List of (word, POS tag) tuples

    Returns:
        dict: Dictionary mapping POS tags to (start, end) integer ranges
    """
    # Step 1: Collect unique words per POS tag
    pos_to_unique_words = defaultdict(set)
    for word, pos_tag in all_tokens:
        pos_to_unique_words[pos_tag].add(word)

    unique_word_counts = {
        pos_tag: len(words) for pos_tag, words in pos_to_unique_words.items()
    }

    # Step 2: Handle 'NNP' allocation
    if "NNP" in unique_word_counts:
        actual_unique_NNP = unique_word_counts["NNP"]
        allocation_for_NNP = min(2000, actual_unique_NNP)
    else:
        allocation_for_NNP = 0

    # Step 3: Process other POS tags
    other_pos_tags = [pos_tag for pos_tag in unique_word_counts if pos_tag != "NNP"]
    sorted_other_pos_tags = sorted(other_pos_tags, key=lambda x: unique_word_counts[x])

    # Calculate total unique words for other POS tags
    total_other_unique = sum(unique_word_counts[pos_tag] for pos_tag in other_pos_tags)

    # Step 4: Calculate scaling factor for other tags
    remaining = MAX_VOCAB_SIZE - 1 - allocation_for_NNP  # Reserve 1 for BOS
    if total_other_unique > remaining and total_other_unique > 0:
        scaling_factor = remaining / total_other_unique
        print(
            f"Scaling non-NNP vocabulary by factor of {scaling_factor:.4f} to fit within limit"
        )
    else:
        scaling_factor = 1.0

    # Step 5: Assign ranges
    tokenizer_dict = {}
    current_index = 0

    # Assign ranges to other POS tags
    for pos_tag in sorted_other_pos_tags:
        num_unique = max(1, int(unique_word_counts[pos_tag] * scaling_factor))
        tokenizer_dict[pos_tag] = (current_index, current_index + num_unique)
        current_index += num_unique

    # Assign range to 'NNP' if present
    if "NNP" in unique_word_counts:
        tokenizer_dict["NNP"] = (current_index, current_index + allocation_for_NNP)
        current_index += allocation_for_NNP

    # Step 6: Finalize vocabulary size
    global vocab_size
    vocab_size = current_index
    print("POS tags and their corresponding integer ranges:")
    for pos_tag, (start, end) in tokenizer_dict.items():
        print(f"{pos_tag}: {start}-{end}")
    print(f"Vocabulary size: {vocab_size:,}")

    if vocab_size > MAX_VOCAB_SIZE - 1:
        raise ValueError(
            f"Vocabulary size {vocab_size} exceeds maximum allowed size of {MAX_VOCAB_SIZE - 1}"
        )

    return tokenizer_dict


def tokenize_pos_tags(pos_tags, tokenizer_dict):
    """
    Tokenizes each part-of-speech tag into a random integer from its assigned range.

    For each POS tag, selects a random integer from the range corresponding to that tag
    in the tokenizer dictionary. This creates an integer representation where the value's
    range indicates the grammatical function.

    Args:
        pos_tags (list): List of part-of-speech tags
        tokenizer_dict (dict): Dictionary mapping POS tags to integer ranges

    Returns:
        list: List of integer tokens representing the input tags
    """
    tokenized = []
    for pos_tag in pos_tags:
        if pos_tag in tokenizer_dict:
            start, end = tokenizer_dict[pos_tag]
            token_id = random.randint(start, end - 1)
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

    # Split into train and validation sets (95/5 split)
    n = len(pos_tags)
    train_pos_tags = pos_tags[: int(n * 0.95)]
    val_pos_tags = pos_tags[int(n * 0.95) :]

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
    for i in range(20):
        print(f"Train: {train_tokenized[i]} -> {train_pos_tags[i]}")

    # Save to binary files for efficient storage and loading using uint16
    train_tokenized = np.array(train_tokenized, dtype=np.uint16)  # Changed to uint16
    val_tokenized = np.array(val_tokenized, dtype=np.uint16)  # Changed to uint16
    train_tokenized.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
    val_tokenized.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

    # Save metadata with vocabulary size for model training
    meta = {
        "vocab_size": vocab_size,  # Number of unique IDs used (0 to vocab_size-1)
        "tokenizer_dict": tokenizer_dict,  # Mapping of POS tags to integer ranges
    }
    with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    main()
