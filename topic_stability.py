"""
topic_stability.py
==================

Measures how STABLE the discovered topics are across independent training
runs (different random seeds). Topic models are non-convex: two runs with
different seeds can find different topics even on identical data. A model
whose topics barely change across seeds is more trustworthy -- its topics
reflect real structure in the corpus, not an accident of initialization.

We quantify this with the Jaccard similarity of top-N word sets. For two
runs A and B, each topic is a set of its top-N words. Because topic k in
run A need not correspond to topic k in run B (topics are unordered and
can be permuted between runs), we align them: for every topic in A we find
its best-matching topic in B (highest Jaccard), average those best matches,
and symmetrise by doing the same from B to A. This "best-match Jaccard" is
a standard way to compare two topic models (e.g. Greene et al. 2014).

    Jaccard(S_a, S_b) = |S_a ∩ S_b| / |S_a ∪ S_b|      in [0, 1]

    stability(A, B) = mean over topics of best-match Jaccard, symmetrised

    overall stability = mean of stability(A, B) over all run pairs

1.0 = every topic reappears identically across seeds (perfectly stable);
0.0 = no topic shares any top word across seeds (completely unstable).

INPUT  per run: <run_dir>/beta_per_year.npy   shape [T, K, V]  (exported by
                evaluate.py) and a shared vocab. We compare topics at the
                LAST time slice by default (most-trained, most words seen),
                but --slice all averages stability across every slice.

USAGE
-----
    python topic_stability.py \
        --run_dirs runs/dsntm_k30_seed14 runs/dsntm_k30_seed2026 runs/dsntm_k30_seed11014 \
        --vocab data/processed/vocab.pkl \
        --topn 10 \
        --out runs/dsntm_k30_stability.json

Prints a pairwise stability matrix and the overall mean, and writes a JSON
report. Purely an analysis script -- it trains nothing and modifies no run.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np


def load_vocab(path):
    """vocab may be a list/array (index->word) or a dict. We only need V,
    so we return its length; the actual words are not required for Jaccard
    (we compare word INDICES, which is equivalent and avoids any ambiguity
    about dict ordering)."""
    with open(path, "rb") as f:
        vocab = pickle.load(f)
    if isinstance(vocab, dict):
        return len(vocab)
    return len(vocab)


def top_n_sets(beta_slice, topn):
    """beta_slice: [K, V] word-distribution per topic at one time slice.
    Returns a list of K python sets, each holding the topn top word indices
    for that topic."""
    K = beta_slice.shape[0]
    sets = []
    for k in range(K):
        # argsort ascending -> take last topn -> those are the largest probs
        top_idx = np.argsort(beta_slice[k])[-topn:]
        sets.append(set(int(i) for i in top_idx))
    return sets


def jaccard(a, b):
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def best_match_stability(sets_a, sets_b):
    """Symmetric best-match Jaccard between two runs' topic sets.
    For each topic in A, find its highest Jaccard with any topic in B;
    average. Repeat B->A. Return the mean of the two directions."""
    def one_direction(src, dst):
        vals = []
        for s in src:
            best = max(jaccard(s, d) for d in dst)
            vals.append(best)
        return float(np.mean(vals))

    a_to_b = one_direction(sets_a, sets_b)
    b_to_a = one_direction(sets_b, sets_a)
    return 0.5 * (a_to_b + b_to_a)


def load_beta(run_dir):
    # DSNTM evaluate.py exports beta_per_time.npy; older D-ETM runs used
    # beta_per_year.npy. Accept either.
    candidates = ["beta_per_time.npy", "beta_per_year.npy"]
    path = None
    for name in candidates:
        cand = os.path.join(run_dir, name)
        if os.path.isfile(cand):
            path = cand
            break
    if path is None:
        raise FileNotFoundError(
            f"none of {candidates} found in {run_dir}. Run evaluate.py on "
            f"{run_dir} first so it exports the per-slice beta.")
    beta = np.load(path)
    if beta.ndim != 3:
        raise ValueError(
            f"{path} has shape {beta.shape}, expected 3D [T, K, V].")
    return beta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dirs", nargs="+", required=True,
                   help="two or more run folders, each containing "
                        "beta_per_year.npy")
    p.add_argument("--vocab", default=None,
                   help="optional vocab.pkl, only used to sanity-check V")
    p.add_argument("--topn", type=int, default=10,
                   help="number of top words per topic to compare (default "
                        "10, matching the coherence protocol)")
    p.add_argument("--slice", default="last",
                   help="'last' (default): compare topics at the final time "
                        "slice. 'all': average stability across every slice. "
                        "an integer: use that slice index.")
    p.add_argument("--out", default=None,
                   help="optional path to write a JSON report")
    args = p.parse_args()

    if len(args.run_dirs) < 2:
        sys.exit("need at least two --run_dirs to measure stability")

    betas = [load_beta(d) for d in args.run_dirs]

    # shape checks: all runs must share T, K, V to be comparable
    shapes = [b.shape for b in betas]
    if len({s for s in shapes}) != 1:
        print("WARNING: run betas have different shapes:")
        for d, s in zip(args.run_dirs, shapes):
            print(f"  {d}: {s}")
        # we can still compare if K and V match; T mismatch only matters
        # for 'all'. Fail only if K or V differ.
        Ks = {s[1] for s in shapes}
        Vs = {s[2] for s in shapes}
        if len(Ks) != 1 or len(Vs) != 1:
            sys.exit("K or V differ across runs -- cannot compare topics.")

    T, K, V = betas[0].shape
    if args.vocab is not None:
        vlen = load_vocab(args.vocab)
        if vlen != V:
            print(f"WARNING: vocab length {vlen} != beta V dim {V}. "
                  f"Proceeding with beta's V={V}.")

    # decide which slice indices to evaluate
    if args.slice == "last":
        slice_indices = [min(b.shape[0] for b in betas) - 1]
    elif args.slice == "all":
        slice_indices = list(range(min(b.shape[0] for b in betas)))
    else:
        slice_indices = [int(args.slice)]

    n = len(args.run_dirs)
    # pairwise stability averaged over the chosen slices
    pair_scores = {}
    matrix = np.full((n, n), np.nan)
    for i in range(n):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            per_slice = []
            for t in slice_indices:
                sa = top_n_sets(betas[i][t], args.topn)
                sb = top_n_sets(betas[j][t], args.topn)
                per_slice.append(best_match_stability(sa, sb))
            score = float(np.mean(per_slice))
            pair_scores[f"{i}-{j}"] = score
            matrix[i, j] = score
            matrix[j, i] = score

    # overall = mean of upper-triangle pair scores
    triu = [matrix[i, j] for i in range(n) for j in range(i + 1, n)]
    overall = float(np.mean(triu))

    # ---- report ----
    names = [os.path.basename(os.path.normpath(d)) for d in args.run_dirs]
    print(f"\nTopic stability (best-match Jaccard, top-{args.topn} words, "
          f"slice={args.slice})")
    print(f"K={K}, V={V}, {n} runs\n")

    w = max(len(x) for x in names) + 2
    header = " " * w + "".join(f"{x:>{w}}" for x in names)
    print(header)
    for i in range(n):
        row = f"{names[i]:<{w}}"
        for j in range(n):
            row += f"{matrix[i, j]:>{w}.3f}"
        print(row)

    print(f"\nOverall mean pairwise stability: {overall:.4f}")
    print("(1.0 = identical topics across seeds, 0.0 = no shared top words)")

    if args.out:
        report = {
            "topn": args.topn,
            "slice": args.slice,
            "K": int(K), "V": int(V),
            "run_dirs": args.run_dirs,
            "run_names": names,
            "pairwise_matrix": matrix.tolist(),
            "pair_scores": pair_scores,
            "overall_mean_stability": overall,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()