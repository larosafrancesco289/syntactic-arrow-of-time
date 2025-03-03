import os
import pickle
import numpy as np
import torch

# ...existing code (if any)...

# Import the GPT model and its config
from model import GPT, GPTConfig
import config.train_shakespeare_pos_id as cfg


def main():
    # Path settings
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data", "shakespeare_pos_id")

    # Load tokenizer meta information
    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    id_to_pos = meta["id_to_pos"]
    vocab_size = meta["vocab_size"]

    # Load a sample of tokenized training data (e.g., first 10 tokens)
    train_bin = os.path.join(data_dir, "train.bin")
    train_tokens = np.fromfile(train_bin, dtype=np.uint32)
    init_tokens = torch.tensor(train_tokens[:10], dtype=torch.long).unsqueeze(
        0
    )  # (batch, seq)

    # Build GPT model with configuration from config file
    config = GPTConfig(
        block_size=cfg.block_size,
        vocab_size=vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=True,  # Ensure bias=True as in train_shakespeare_pos_id.py
    )
    model = GPT(config)
    model.eval()  # Set to evaluation mode

    # Generate additional tokens (for instance, 20 more tokens)
    with torch.no_grad():
        generated = model.generate(
            init_tokens, max_new_tokens=20, temperature=1.0, top_k=10
        )
    generated = generated[0].tolist()  # Convert to list

    # Map generated token IDs back to POS tags using id_to_pos dictionary
    generated_pos = [id_to_pos.get(token, "UNK") for token in generated]
    print("Generated POS tags:")
    print(generated_pos)


if __name__ == "__main__":
    main()
