"""
train_embeddings.py 
Trains 300-d skip-gram Word2Vec embeddings from scratch on the TRAIN split
(train_docs.txt produced by prepare_data.py).

Every vocabulary word is guaranteed a vector: train_docs.txt only contains
in-vocab tokens and min_count=1 is used (the corpus-level min_df=10 filter
already happened in prepare_data.py). The output is a dense float32 matrix
aligned with vocab.json row-for-row.

Usage
-----
    python train_embeddings.py --data_dir data/processed --epochs 10
"""
import argparse
import json
import os

import numpy as np
from gensim.models import Word2Vec


class LineCorpus:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path) as f:
            for line in f:
                toks = line.split()
                if toks:
                    yield toks


def main(args):
    with open(os.path.join(args.data_dir, "vocab.json")) as f:
        vocab = json.load(f)
    corpus = LineCorpus(os.path.join(args.data_dir, "train_docs.txt"))

    print(f"Training Word2Vec skip-gram: dim={args.dim}, window={args.window}, "
          f"negative={args.negative}, epochs={args.epochs}")
    model = Word2Vec(
        sentences=corpus,
        vector_size=args.dim,
        sg=1,
        window=args.window,
        negative=args.negative,
        min_count=1,
        workers=args.workers,
        epochs=args.epochs,
        seed=args.seed,
    )

    emb = np.zeros((len(vocab), args.dim), dtype=np.float32)
    missing = 0
    for i, w in enumerate(vocab):
        if w in model.wv:
            emb[i] = model.wv[w]
        else:  
            emb[i] = np.random.RandomState(i).normal(0, 0.1, args.dim)
            missing += 1
    out = os.path.join(args.data_dir, "embeddings_300d.npy")
    np.save(out, emb)
    print(f"Saved {out}  shape={emb.shape}  missing={missing}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/processed")
    p.add_argument("--dim", type=int, default=300)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--negative", type=int, default=10)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=2019)
    main(p.parse_args())
