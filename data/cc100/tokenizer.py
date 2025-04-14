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
import lzma

# Maximum vocabulary size for 16-bit representation
MAX_VOCAB_SIZE = 65535  # 2^16 - 1

# Global variable to store the total vocabulary size
vocab_size = 0


# Remove or comment out the entire load_openwebtext_subset function


def load_cc100_subset(file_path, target_size_gb=1):
    """
    Loads a subset of the local CC100 dataset (en.txt.xz).
    Returns the raw text as a single string.

    Args:
        file_path (str): The path to the en.txt.xz file.
        target_size_gb (float): The approximate target size in gigabytes.

    Returns:
        str: The loaded text data as a single string, or empty string on error.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return ""

    target_size_bytes = target_size_gb * 10**9
    text_data = []
    total_size_bytes = 0
    chunk_size = 1024 * 1024  # Read 1MB chunks (of characters) at a time

    print(f"Loading CC100 subset from {file_path}...")
    try:
        with lzma.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
            with tqdm(
                total=target_size_bytes, unit="B", unit_scale=True, desc="Reading CC100"
            ) as pbar:
                while total_size_bytes < target_size_bytes:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break  # End of file

                    # Estimate chunk byte size (encoding needed for accuracy)
                    # For performance, we can approximate or calculate precisely
                    chunk_bytes = len(chunk.encode("utf-8", errors="ignore"))

                    # Check if adding the full chunk exceeds the target
                    if total_size_bytes + chunk_bytes > target_size_bytes:
                        # Estimate how much of the chunk to keep
                        remaining_bytes = target_size_bytes - total_size_bytes
                        # Calculate approximate character fraction
                        # This is an estimation as char != byte in UTF-8
                        fraction_to_keep = (
                            remaining_bytes / chunk_bytes if chunk_bytes > 0 else 0
                        )
                        chars_to_keep = max(0, int(len(chunk) * fraction_to_keep))
                        # Take a bit less initially to be safe, then refine
                        estimated_chars = max(
                            1, chars_to_keep - 10
                        )  # Adjust buffer as needed
                        chunk = chunk[:estimated_chars]

                        # Refine by checking actual byte size iteratively
                        chunk_bytes = len(chunk.encode("utf-8", errors="ignore"))
                        while (
                            total_size_bytes + chunk_bytes < target_size_bytes
                            and estimated_chars < len(f.buffer)
                        ):  # check if more chars available
                            chunk += f.buffer[estimated_chars]
                            estimated_chars += 1
                            chunk_bytes = len(chunk.encode("utf-8", errors="ignore"))

                        # Final trim if we overshot
                        while (
                            total_size_bytes + chunk_bytes > target_size_bytes
                            and len(chunk) > 0
                        ):
                            chunk = chunk[:-1]  # Remove last character
                            chunk_bytes = len(chunk.encode("utf-8", errors="ignore"))

                    text_data.append(chunk)
                    actual_bytes_added = chunk_bytes
                    total_size_bytes += actual_bytes_added
                    pbar.update(actual_bytes_added)

                    if total_size_bytes >= target_size_bytes:
                        # Ensure the progress bar reflects completion if we hit the target exactly or slightly overshot before truncation
                        pbar.n = target_size_bytes
                        pbar.last_print_n = target_size_bytes
                        pbar.refresh()
                        break  # Target reached

    except lzma.LZMAError as e:
        print(f"Error decompressing file: {e}")
        return ""
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""

    final_gb = total_size_bytes / 10**9
    print(
        f"\nLoaded approximately {final_gb:.2f} GB of text (target was {target_size_gb} GB)"
    )
    # Join the chunks into a single string.
    # Assumes CC100 uses double newlines internally like OpenWebText examples.
    # If not, the split in extract_pos_tags might behave differently.
    return "".join(text_data)


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
    1. Read the CC100 dataset
    2. Extract part-of-speech tags using spaCy
    3. Create a tokenizer dictionary mapping POS tags to integer ranges
    4. Split data into training and validation sets
    5. Convert POS tags to integer tokens
    6. Save tokenized data and metadata to disk
    """
    # Define the path to your local CC100 file
    cc100_file_path = "path/to/your/en.txt.xz"

    # Target size in GB for the subset to load
    target_gb = 1

    # Load CC100
    print(f"Attempting to load data from: {os.path.abspath(cc100_file_path)}")
    data = load_cc100_subset(cc100_file_path, target_size_gb=target_gb)

    if not data:  # Handle case where loading failed (e.g., file not found)
        print(
            "Failed to load data. Please check the file path and ensure it's readable."
        )
        print(f"Looked for: {os.path.abspath(cc100_file_path)}")
        return  # Exit if data loading failed

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
