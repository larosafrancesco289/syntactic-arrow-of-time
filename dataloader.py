import numpy as np
import torch
from torch.utils.data import Dataset


class POSDataset(Dataset):
    """
    PyTorch Dataset for part-of-speech tokenized data, supporting both forward and backward sequences.
    """

    def __init__(self, file_path, block_size, vocab_size, stride=1, backwards=False):
        """
        Args:
            file_path (str): Path to binary file with tokenized data
            block_size (int): Number of tokens in a sequence
            stride (int): Step size between consecutive sequences
            backwards (bool): Whether to return sequences in reverse order
        """
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.stride = stride
        self.backwards = backwards
        self.data = np.memmap(file_path, dtype=np.uint16, mode="r")

        # Validate stride
        if stride < 1:
            raise ValueError("Stride must be a positive integer")

    def __len__(self):
        """
        Returns the number of sequences available given the stride and block size.
        """
        return (len(self.data) - self.block_size) // self.stride + 1

    def __getitem__(self, idx):
        """
        Retrieves a sequence starting at idx * stride.
        """
        start_idx = idx * self.stride
        chunk = self.data[
            start_idx : start_idx + self.block_size
        ]  # Shape: (block_size,)

        if self.backwards:
            # Backward direction: append BOS, reverse, then create input and target
            chunk_bos = np.append(chunk, self.vocab_size)  # Shape: (block_size + 1,)
            chunk_bos = chunk_bos[::-1]  # Reverse the sequence
            x = chunk_bos[:-1]  # Input: first block_size elements
            y = chunk_bos[1:]  # Target: last block_size elements
        else:
            # Forward direction: prepend BOS, then create input and target
            chunk_bos = np.insert(chunk, 0, self.vocab_size)  # Shape: (block_size + 1,)
            x = chunk_bos[:-1]  # Input: first block_size elements
            y = chunk_bos[1:]  # Target: last block_size elements

        # Convert to PyTorch tensors
        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))

        return x, y
