"""
umass_coherence.py
==================

Computes UMass coherence for DSNTM topics.

UMass (Mimno et al. 2011) measures topic coherence by counting how many
documents contain pairs of the topic's top words. High = coherent topics.

USAGE
-----
    python umass_coherence.py \
        --run_dir runs/dsntm_k30 \
        --corpus_dir data/processed \
        --topn 10 --slice last
"""
import argparse
import json
import os
import numpy as np
from scipy import sparse


def load_beta(run_dir):
    """Load beta_per_time.npy [T, K, V]."""
    candidates = ["beta_per_time.npy", "beta_per_year.npy"]
    path = None
    for name in candidates:
        cand = os.path.join(run_dir, name)
        if os.path.isfile(cand):
            path = cand
            break
    if path is None:
        raise FileNotFoundError(f"none of {candidates} found in {run_dir}")
    return np.load(path)


def umass_coherence_topic(top_word_ids, corpus_slice):
    """UMass coherence for one topic.
    
    Args:
        top_word_ids: array of top N word indices
        corpus_slice: [num_docs, V] sparse or dense doc-term matrix
    
    Returns:
        float: UMass score (typically -20 to 0, higher = better)
    """
    top_word_ids = np.array(top_word_ids, dtype=int)
    N = len(top_word_ids)
    if N < 2:
        return 0.0
    
    # Extract columns for top words
    if sparse.issparse(corpus_slice):
        subset = corpus_slice[:, top_word_ids].toarray()
    else:
        subset = corpus_slice[:, top_word_ids]
    
    scores = []
    for i in range(N):
        for j in range(i):
            # Count docs containing both word i and j
            docs_ij = (subset[:, i] > 0) & (subset[:, j] > 0)
            # Count docs containing word j
            docs_j = (subset[:, j] > 0)
            
            if docs_j.sum() > 0:
                # UMass: log((count_both + 1) / count_j)
                score = np.log((docs_ij.sum() + 1.0) / docs_j.sum())
                scores.append(score)
    
    return float(np.mean(scores)) if scores else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True,
                   help="run folder with beta_per_time.npy")
    p.add_argument("--corpus_dir", default="data/processed",
                   help="directory with train_bow.npz + train_times.npy")
    p.add_argument("--topn", type=int, default=10,
                   help="top N words per topic (default 10)")
    p.add_argument("--slice", default="last",
                   help="'last', 'all', or slice index")
    p.add_argument("--out", default=None,
                   help="optional JSON output")
    args = p.parse_args()
    
    # Load beta
    beta = load_beta(args.run_dir)
    T, K, V = beta.shape
    
    # Decide slices
    if args.slice == "last":
        slice_indices = [T - 1]
    elif args.slice == "all":
        slice_indices = list(range(T))
    else:
        slice_indices = [int(args.slice)]
    
    print(f"\nUMass coherence: K={K}, V={V}, topn={args.topn}\n")
    
    # Load corpus (single file format: train_bow.npz + train_times.npy)
    bow_path = os.path.join(args.corpus_dir, "train_bow.npz")
    times_path = os.path.join(args.corpus_dir, "train_times.npy")
    
    if not os.path.isfile(bow_path) or not os.path.isfile(times_path):
        raise FileNotFoundError(
            f"Missing {bow_path} or {times_path}")
    
    bow_full = sparse.load_npz(bow_path)
    times_full = np.load(times_path)
    print(f"loaded train_bow.npz {bow_full.shape} + train_times.npy\n")
    
    # Compute UMass
    umass_per_slice = {}
    all_scores = []
    
    for t in slice_indices:
        print(f"slice {t}:")
        
        # Split corpus by time slice
        mask = times_full == t
        corpus_t = bow_full[mask]
        
        slice_scores = []
        for k in range(K):
            top_idx = np.argsort(beta[t, k])[-args.topn:][::-1]
            score = umass_coherence_topic(top_idx, corpus_t)
            slice_scores.append(score)
            all_scores.append(score)
        
        mean_umass = np.mean(slice_scores)
        umass_per_slice[t] = mean_umass
        print(f"  UMass (mean over {K} topics): {mean_umass:.4f}")
    
    overall_mean = float(np.mean(all_scores)) if all_scores else 0.0
    print(f"\nOverall mean UMass: {overall_mean:.4f}")
    print("(range -20 to 0; higher = more coherent)")
    
    # JSON output
    if args.out:
        report = {
            "topn": args.topn,
            "slice_indices": slice_indices,
            "K": int(K),
            "V": int(V),
            "umass_per_slice": {str(t): v for t, v in umass_per_slice.items()},
            "overall_mean_umass": overall_mean,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()