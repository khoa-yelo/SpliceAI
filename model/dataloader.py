"""
Dataset class for SpliceAI model.
"""

import pickle
from torch.utils.data import Dataset
import torch
import numpy as np


class SpliceDataset(Dataset):
    """
    Dataset class for SpliceAI model.
    Args:
        path (str): Path to the dataset file. (pickle file)
        frac (float): Fraction of the dataset to use.
        flank (int): Flank size for training data.
    """

    def __init__(self, path, frac, flank: int):
        with open(path, "rb") as f:
            self.data = pickle.load(f)
        # choose random subset of data
        if frac < 1.0:
            self.data = np.random.choice(
                self.data, int(len(self.data) * frac), replace=False
            )
        self.flank = flank
        self.desired_len = 5000 + 2 * flank

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Input: (L, 4) → (4, L)
        seq = torch.tensor(item["sequence"], dtype=torch.float32).permute(1, 0)
        # Target: (L, C) → (L,) with class indices
        tgt = torch.tensor(item["y"], dtype=torch.float32).argmax(dim=1)

        L = seq.size(1)
        if L < self.desired_len:
            raise ValueError(
                f"Sequence length {L} is shorter than desired {self.desired_len}"
            )

        # compute cropping bounds
        trim = (L - self.desired_len) // 2
        start = trim
        end = trim + self.desired_len

        seq = seq[:, start:end]

        return seq, tgt
