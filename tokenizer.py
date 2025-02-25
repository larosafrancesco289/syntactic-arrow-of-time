import os
import requests
import spacy
from tqdm import tqdm
from collections import Counter


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
    # Initialize spaCy with English model
    nlp = spacy.load("en_core_web_sm")
    # Increase the max length to handle large texts
    nlp.max_length = 2_000_000

    # Process the text
    doc = nlp(text)

    # Extract POS tags
    pos_list = []
    print("Extracting part-of-speech tags...")
    for token in tqdm(doc):
        # We only keep the POS label, not the actual word
        pos_list.append(token.tag_)

    return pos_list


def main():
    """
    Main function that coordinates the tokenization process.
    """
    # Get input data
    input_path = download_dataset()
    data = read_data(input_path)

    # Extract POS tags
    pos_tags = extract_pos_tags(data)

    # Count occurrences of each POS tag
    pos_counts = Counter(pos_tags)

    # Display results
    print("\nPOS Tag Distribution:")
    print(pos_counts)

    # You could add more analysis here
    print(f"Total tokens processed: {len(pos_tags)}")
    print(f"Unique POS tags found: {len(pos_counts)}")


if __name__ == "__main__":
    main()
