"""
It builds the DSNTM dataset from the raw SciEvo pickle, using preprocessing.
For Input 
    scievo_cscl_cslg_2014_2021.pkl
        {'papers': DataFrame[id, title, abstract, year, month, tags],
         'references': DataFrame (empty, unused here)}

Preprocessing decisions 
1.  text = title + " " + abstract, lowercased; punctuation and digits stripped;
    tokens of length <= 1 dropped.
2.  Stopwords: stops.txt plus the EXTRA_STOP set below.
3.  2021 papers are held out COMPLETELY and saved separately for T+1 forecast validation.
4.  Vocabulary built on 2014-2020 with CountVectorizer(min_df=10, max_df=0.70).
5.  Random (seeded) 85/10/5 train/valid/test split at the document level.
6.  After splitting, the vocabulary is restricted to words that occur in the TRAIN split; 
    documents that become empty after this restriction are dropped.
7.  Test set is additionally split into h1/h2 halves (first half of the token
    sequence / second half) for document-completion perplexity.
8.  Time slices: one per year, 2014..2020 -> t = 0..6 (T = 7).

Output layout (out_dir, default data/processed/)
------------------------------------------------
    vocab.json                 list[str], index = word id
    <split>_bow.npz            scipy CSR [num_docs, V] raw counts
    <split>_times.npy          int64 [num_docs] time-slice index 0..T-1
    test_h1_bow.npz / test_h2_bow.npz   completion halves (same rows as test)
    holdout2021_bow.npz        2021 docs in the same vocab (empty docs dropped)
    dataset_meta.json          sizes, thresholds, per-year counts, etc.
    train_docs.txt             one tokenized train doc per line (for Word2Vec
                               and as the NPMI reference corpus)

Usage
-----
    python prepare_data.py --pkl scievo_cscl_cslg_2014_2021.pkl \
        --out_dir data/processed --seed 2019
"""
import argparse
import json                                               
import os
import pickle
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
 
EXTRA_STOP = {
    "paper", "papers", "propose", "proposed", "approach", "approaches",
    "method", "methods", "result", "results", "show", "shows", "based",
    "using", "used", "use", "new", "novel", "present", "demonstrate",
    "task", "tasks", "model", "models", "data", "dataset", "datasets",
    "state", "art", "performance",
}

TOKEN_RE = re.compile(r"[a-z]{2,}")  # letters only, length >= 2

FIRST_YEAR, LAST_TRAIN_YEAR, HOLDOUT_YEAR = 2014, 2020, 2021


def load_stopwords(stops_path):
    with open(stops_path) as f:
        stops = {w.strip() for w in f if w.strip()}
    return stops | EXTRA_STOP


def clean_text(title, abstract):
    """Lowercase, strip punctuation/digits (anything not [a-z] acts as a
    separator), keep tokens of length >= 2. Stopwords are removed later by
    CountVectorizer / the tokenizer wrapper so that exactly one tokenizer
    definition exists."""
    text = f"{title} {abstract}".lower()
    return TOKEN_RE.findall(text)


def docs_to_bow(token_docs, word2id):
    """Token lists -> CSR count matrix restricted to word2id."""
    indptr, indices, values = [0], [], []
    for toks in token_docs:
        counts = {}
        for w in toks:
            wid = word2id.get(w)
            if wid is not None:
                counts[wid] = counts.get(wid, 0) + 1
        indices.extend(counts.keys())
        values.extend(counts.values())
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (np.array(values, dtype=np.int64),
         np.array(indices, dtype=np.int64),
         np.array(indptr, dtype=np.int64)),
        shape=(len(token_docs), len(word2id)),
    )


def split_halves(token_docs, word2id):
    """D-ETM-style completion split: first half of the (in-vocab) token
    sequence vs. second half."""
    h1_docs, h2_docs = [], []
    for toks in token_docs:
        iv = [w for w in toks if w in word2id]
        mid = len(iv) // 2
        h1_docs.append(iv[:mid])
        h2_docs.append(iv[mid:])
    return docs_to_bow(h1_docs, word2id), docs_to_bow(h2_docs, word2id)


def main(args):
    rng = np.random.RandomState(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    stops = load_stopwords(args.stops)
    print(f"Loaded {len(stops)} stopwords (stops.txt + EXTRA_STOP)")

    with open(args.pkl, "rb") as f:
        raw = pickle.load(f)
    papers = raw["papers"].copy()
    papers = papers[(papers["year"] >= FIRST_YEAR) & (papers["year"] <= HOLDOUT_YEAR)]
    print(f"Loaded {len(papers)} papers, years "
          f"{papers['year'].min()}-{papers['year'].max()}")

    print("Tokenizing (this is the only tokenizer in the pipeline)...")
    papers["tokens"] = [
        [w for w in clean_text(t, a) if w not in stops]
        for t, a in zip(papers["title"].fillna(""), papers["abstract"].fillna(""))
    ]

    main_df = papers[papers["year"] <= LAST_TRAIN_YEAR].reset_index(drop=True)
    hold_df = papers[papers["year"] == HOLDOUT_YEAR].reset_index(drop=True)
    print(f"2014-2020: {len(main_df)} docs | 2021 holdout: {len(hold_df)} docs")
 
    # CountVectorizer over pre-tokenized docs; min_df counts documents.
    cv = CountVectorizer(
        min_df=args.min_df, max_df=args.max_df,
        tokenizer=lambda x: x, preprocessor=lambda x: x, lowercase=False,
        token_pattern=None,
    )
    cv.fit(main_df["tokens"].tolist())
    vocab_stage1 = set(cv.vocabulary_.keys())
    print(f"Stage-1 vocab (min_df={args.min_df}, max_df={args.max_df}): "
          f"{len(vocab_stage1)}")

    #  85/10/5 random document-level split  
    n = len(main_df)
    perm = rng.permutation(n)
    n_train = int(round(0.85 * n))
    n_valid = int(round(0.10 * n))
    idx = {"train": perm[:n_train],
           "valid": perm[n_train:n_train + n_valid],
           "test": perm[n_train + n_valid:]}
    print(f"Split sizes (pre-filter): train={len(idx['train'])} "
          f"valid={len(idx['valid'])} test={len(idx['test'])}")

    #  Restrict vocab to words present in TRAIN (D-ETM script order)  
    train_words = set()
    for i in idx["train"]:
        train_words.update(w for w in main_df["tokens"][i] if w in vocab_stage1)
    vocab = sorted(vocab_stage1 & train_words)
    word2id = {w: i for i, w in enumerate(vocab)}
    print(f"Final vocab (restricted to train occurrences): {len(vocab)}")
    if (args.min_df, args.max_df) == (10, 0.70) and abs(len(vocab) - 15282) > 1500:
        print("WARNING: vocab size differs from the prior D-ETM run's 15,282 "
              "by more than ~10% -- investigate preprocessing parity before "
              "comparing models.")

    meta = {
        "vocab_size": len(vocab),
        "min_df": args.min_df, "max_df": args.max_df,
        "seed": args.seed,
        "num_times": LAST_TRAIN_YEAR - FIRST_YEAR + 1,
        "years": list(range(FIRST_YEAR, LAST_TRAIN_YEAR + 1)),
        "holdout_year": HOLDOUT_YEAR,
        "extra_stopwords": sorted(EXTRA_STOP),
        "splits": {},
    }

    def build_split(name, indices, df):
        toks = [df["tokens"][i] for i in indices]
        yrs = np.array([df["year"][i] for i in indices], dtype=np.int64)
        bow = docs_to_bow(toks, word2id)
        keep = np.asarray(bow.sum(axis=1)).ravel() > 1  # drop docs with <2 in-vocab tokens
        bow, yrs = bow[keep], yrs[keep]
        toks = [t for t, k in zip(toks, keep) if k]
        times = yrs - FIRST_YEAR
        sparse.save_npz(os.path.join(args.out_dir, f"{name}_bow.npz"), bow)
        np.save(os.path.join(args.out_dir, f"{name}_times.npy"), times)
        meta["splits"][name] = {
            "num_docs": int(bow.shape[0]),
            "dropped_short": int((~keep).sum()),
            "per_year": {int(y): int((yrs == y).sum()) for y in np.unique(yrs)},
        }
        print(f"{name}: {bow.shape[0]} docs ({(~keep).sum()} dropped as empty/short)")
        return toks

    train_toks = build_split("train", idx["train"], main_df)
    build_split("valid", idx["valid"], main_df)
    test_toks = build_split("test", idx["test"], main_df)
    build_split("holdout2021", np.arange(len(hold_df)), hold_df)

    # completion halves for the (already filtered) test docs
    h1, h2 = split_halves(test_toks, word2id)
    sparse.save_npz(os.path.join(args.out_dir, "test_h1_bow.npz"), h1)
    sparse.save_npz(os.path.join(args.out_dir, "test_h2_bow.npz"), h2)

    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    with open(os.path.join(args.out_dir, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # reference corpus for Word2Vec + NPMI (train docs, in-vocab tokens only)
    with open(os.path.join(args.out_dir, "train_docs.txt"), "w") as f:
        for toks in train_toks:
            f.write(" ".join(w for w in toks if w in word2id) + "\n")

    print("Done. Wrote", args.out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", type=str, default="scievo_cscl_cslg_2014_2021.pkl")
    p.add_argument("--out_dir", type=str, default="data/processed")
    p.add_argument("--stops", type=str, default="stops.txt")
    p.add_argument("--min_df", type=int, default=10)
    p.add_argument("--max_df", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=2019)
    main(p.parse_args())
