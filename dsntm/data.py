"""
dsntm/data.py 
Loading and batching for the processed dataset. 
"""
import json
import os

import numpy as np
import torch
from scipy import sparse


class Split:
    def __init__(self, data_dir, name, device):
        self.bow = sparse.load_npz(os.path.join(data_dir, f"{name}_bow.npz")) \
            .astype(np.float32)
        self.times = np.load(os.path.join(data_dir, f"{name}_times.npy"))
        self.num_docs = self.bow.shape[0]
        self.device = device

    def batch(self, indices):
        bows = torch.from_numpy(self.bow[indices].toarray()).to(self.device)
        times = torch.from_numpy(self.times[indices]).to(self.device)
        return bows, times


def load_dataset(data_dir, device):
    with open(os.path.join(data_dir, "vocab.json")) as f:
        vocab = json.load(f)
    with open(os.path.join(data_dir, "dataset_meta.json")) as f:
        meta = json.load(f)
    splits = {name: Split(data_dir, name, device)
              for name in ["train", "valid", "test", "holdout2021"]}
              
    # completion halves share the row order of `test`
    for h in ["test_h1", "test_h2"]:
        s = Split.__new__(Split)
        s.bow = sparse.load_npz(os.path.join(data_dir, f"{h}_bow.npz")) \
            .astype(np.float32)
        s.times = splits["test"].times
        s.num_docs = s.bow.shape[0]
        s.device = device
        splits[h] = s
    emb_path = os.path.join(data_dir, "embeddings_300d.npy")
    embeddings = torch.from_numpy(np.load(emb_path)) \
        if os.path.exists(emb_path) else None
    return vocab, meta, splits, embeddings


def get_rnn_input(split, num_times):
    """Per-time-slice mean bag-of-words [T, V] (input to the eta LSTM).
    Deterministic, computed with sparse ops (the official code arrives at the
    same quantity through batched accumulation)."""
    V = split.bow.shape[1]
    out = np.zeros((num_times, V), dtype=np.float32)
    for t in range(num_times):
        mask = split.times == t
        if mask.any():
            out[t] = np.asarray(split.bow[mask].mean(axis=0)).ravel()
    return torch.from_numpy(out).to(split.device)
