import os
import requests
import spacy
from tqdm import tqdm
from collections import defaultdict
import random
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset

# Global variable to store the total vocabulary size
vocab_size = 0


class POSDataset(Dataset):
    """
    PyTorch Dataset for part-of-speech tokenized data, supporting both forward and backward sequences.
    """

    def __init__(self, file_path, block_size, backwards=False):
        """
        Args:
            file_path (str): Path to binary file with tokenized data
            block_size (int): Number of tokens in a sequence
            backwards (bool): Whether to return sequences in reverse order
        """
        self.block_size = block_size
        self.backwards = backwards

        # Load data using memmap to handle large files efficiently
        self.data = np.memmap(file_path, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        if self.backwards:
            # For reverse-direction model: predict previous tokens from future context
            x = self.data[idx + 1 : idx + 1 + self.block_size][::-1]
            y = self.data[idx : idx + self.block_size][::-1]
        else:
            # For forward-direction model: predict next tokens from past context
            x = self.data[idx : idx + self.block_size]
            y = self.data[idx + 1 : idx + 1 + self.block_size]

        # Convert NumPy arrays to PyTorch tensors
        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))

        return x, y


def download_dataset():
    """
    Downloads the tiny Shakespeare dataset if not locally available.

    The dataset is fetched from Karpathy's char-rnn repository.

    Returns:
        str: Path to the downloaded input file
    """
    input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
    if not os.path.exists(input_file_path):
        print("Downloading tiny Shakespeare dataset...")
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w", encoding="utf-8") as f:
            f.write(requests.get(data_url).text)
        print("Download complete.")
    return input_file_path


def read_data(file_path):
    """
    Reads the content of the input file.

    Args:
        file_path (str): Path to the text file to read

    Returns:
        str: Complete content of the file as a string
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


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
    nlp.max_length = 2_000_000  # Increase max length to handle large texts
    print("Active pipeline components (pipe_names):", nlp.pipe_names)
    chunks = [
        chunk.strip() for chunk in text.split("\n\n") if chunk.strip()
    ]  # Split text into chunks separated by double newlines
    all_tokens = []
    print("Extracting words and part-of-speech tags...")
    # Process each chunk through the NLP pipeline with n_process=-1 to use all available CPU cores
    for doc in tqdm(nlp.pipe(chunks, n_process=4), total=len(chunks)):
        all_tokens.extend([(token.text, token.tag_) for token in doc])
    return all_tokens


def create_tokenizer_dict(all_tokens):
    """
    Creates a dictionary mapping POS tags to unique integer ranges.

    For each part-of-speech tag, assigns a continuous range of integers
    based on the number of unique words with that tag. POS tags with more
    unique words are assigned wider ranges.

    Args:
        all_tokens (list): List of (word, POS tag) tuples

    Returns:
        dict: Dictionary mapping POS tags to (start, end) integer ranges
    """
    pos_to_unique_words = defaultdict(set)
    for word, pos_tag in all_tokens:
        pos_to_unique_words[pos_tag].add(word)

    unique_word_counts = {
        pos_tag: len(words) for pos_tag, words in pos_to_unique_words.items()
    }

    # Sort POS tags by number of unique words, descending
    sorted_pos_tags = sorted(
        unique_word_counts, key=unique_word_counts.get, reverse=False
    )

    tokenizer_dict = {}
    current_index = 0
    for pos_tag in sorted_pos_tags:
        num_unique = unique_word_counts[pos_tag]
        tokenizer_dict[pos_tag] = (current_index, current_index + num_unique)
        current_index += num_unique

    print("POS tags and their corresponding integer ranges:")
    for pos_tag, (start, end) in tokenizer_dict.items():
        print(f"{pos_tag}: {start}-{end}")

    global vocab_size
    vocab_size = current_index
    print(f"Vocabulary size: {vocab_size:,}")

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
    1. Download and read the Shakespeare dataset
    2. Extract part-of-speech tags using spaCy
    3. Create a tokenizer dictionary mapping POS tags to integer ranges
    4. Split data into training and validation sets
    5. Convert POS tags to integer tokens
    6. Save tokenized data and metadata to disk
    """
    input_path = download_dataset()
    data = read_data(input_path)

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

    # Save to binary files for efficient storage and loading using uint16
    train_tokenized = np.array(train_tokenized, dtype=np.uint16)
    val_tokenized = np.array(val_tokenized, dtype=np.uint16)
    train_tokenized.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
    val_tokenized.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

    # Save metadata with vocabulary size for model training
    meta = {"vocab_size": vocab_size}
    with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    main()
