import os
import requests
import spacy
from tqdm import tqdm
from collections import Counter
import random
import numpy as np
import pickle

# Store vocab_size for later use
vocab_size = 0


def download_dataset():
    """
    Downloads the tiny Shakespeare dataset if it doesn't exist locally.
    Returns the path to the input file.
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
        file_path (str): Path to the input file

    Returns:
        str: Content of the file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_pos_tags(text):
    """
    Extracts part-of-speech tags from the input text using spaCy.

    Args:
        text (str): Input text to analyze

    Returns:
        list: Part-of-speech tags for each token
    """
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000  # Increase the max length to handle large texts
    doc = nlp(text)

    pos_list = []
    print("Extracting part-of-speech tags...")
    for token in tqdm(doc):
        pos_list.append(token.tag_)  # We only keep the POS label, not the actual word

    return pos_list


def create_tokenizer_dict(pos_list):
    """
    Creates a dictionary mapping POS tags to unique integer ranges.

    Args:
        pos_list (list): List of part-of-speech tags

    Returns:
        dict: Dictionary mapping POS tags to integer ranges
    """
    pos_counts = Counter(pos_list)
    pos_counts = dict(
        sorted(pos_counts.items(), key=lambda item: item[1], reverse=True)
    )

    tokenizer_dict = {}
    current_index = 0
    for pos, count in pos_counts.items():
        tokenizer_dict[pos] = (current_index, current_index + count)
        current_index += count

    # Vocabulary size calculation
    global vocab_size
    max_token_id = current_index - 1  # e.g. 120k if that's how many tokens you had
    vocab_size = max_token_id + 1  # must be one more than the largest ID
    print(f"Vocabulary size: {vocab_size:,}")

    return tokenizer_dict


def tokenize_text(text, tokenizer_dict):
    """
    Tokenizes text by replacing each word with a random number from the range
    corresponding to its POS tag.

    Args:
        text (str): Input text to tokenize
        tokenizer_dict (dict): Dictionary mapping POS tags to integer ranges

    Returns:
        list: List of integer tokens
    """
    tokenized_text = []
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000  # Increase max length to handle large texts

    print("Tokenizing text...")
    doc = nlp(text)

    for token in tqdm(doc):
        pos_tag = token.tag_
        if pos_tag in tokenizer_dict:
            start, end = tokenizer_dict[pos_tag]
            token_id = random.randint(start, end - 1)
            tokenized_text.append(token_id)
        else:
            print(f"Warning: Unknown POS tag '{pos_tag}' for token '{token.text}'")
            tokenized_text.append(-1)  # Use -1 for unknown tags

    return tokenized_text


def main():
    """
    Main function that coordinates the tokenization process.
    """
    input_path = download_dataset()
    data = read_data(input_path)

    pos_tags = extract_pos_tags(data)
    tokenizer_dict = create_tokenizer_dict(pos_tags)

    print(tokenizer_dict)

    # Let's split into train and validation sets
    train_data = data[: int(len(data) * 0.9)]
    val_data = data[int(len(data) * 0.9) :]

    # Tokenize training and validation data
    train_tokenized = tokenize_text(train_data, tokenizer_dict)
    val_tokenized = tokenize_text(val_data, tokenizer_dict)

    # Print the number of tokens in each set
    print(f"Train has {len(train_tokenized):,} tokens")
    print(f"Val has {len(val_tokenized):,} tokens")

    # Convert to numpy arrays and save to binary files
    train_tokenized = np.array(train_tokenized, dtype=np.uint32)
    val_tokenized = np.array(val_tokenized, dtype=np.uint32)
    train_tokenized.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
    val_tokenized.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

    # Save the meta information as well, to help us encode/decode later
    meta = {"vocab_size": vocab_size}
    with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    main()
