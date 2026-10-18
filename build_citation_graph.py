"""
build_citation_graph.py
=======================
Builds the internal citation graph for DSNTM citation regularization,
using SciEvo's HuggingFace datasets (no API calls needed).

Three inputs:
  1. SciEvo "references" config  → arXivId -> [cited S2 paperIds]
  2. SciEvo "semantic_scholar" config → S2 paperId <-> arXivId (bridge)
  3. Your corpus pkl → the 111k arXiv IDs that define "internal"

Output (saved to --out_dir):
  citation_graph.pkl
    {
      "edges":       list of (citing_arxiv_id, cited_arxiv_id) tuples
      "num_edges":   int
      "num_papers_with_citations": int
      "coverage":    fraction of corpus papers that have >=1 internal citation
    }

  citation_matrices.pkl  (produced by --build_matrices, needs processed data)
    {
      "mat_cit":      mat_cit[t][s] = scipy.sparse CSR [docs_at_t, docs_at_s]
      "bow_articles": list of dense normalized BOW tensors per time slice
      "doc_ids_per_slice": list of arXiv IDs per slice (for debugging)
    }

Usage
-----
  # Step 1: build the raw edge list (takes ~5 min, mostly HF loading)
  python build_citation_graph.py \
      --pkl scievo_cscl_cslg_2014_2021.pkl \
      --out_dir data/citations

  # Step 2: build the time-sliced matrices for DSNTM training
  python build_citation_graph.py \
      --pkl scievo_cscl_cslg_2014_2021.pkl \
      --out_dir data/citations \
      --build_matrices \
      --data_dir data/processed
"""
import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np


def clean_arxiv_id(raw_id):
    """Normalize arXiv IDs to the short form used in your corpus.
    Handles: 'http://arxiv.org/abs/1706.03762' -> '1706.03762'
             'cs/9301111' -> 'cs/9301111'
             '1706.03762' -> '1706.03762'
             '1706.03762v1' -> '1706.03762'
    """
    if raw_id is None:
        return None
    s = str(raw_id).strip()
    # strip URL prefix
    for prefix in ["http://arxiv.org/abs/", "https://arxiv.org/abs/"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    # strip version suffix (v1, v2, ...)
    if s and s[-1].isdigit():
        parts = s.rsplit("v", 1)
        if len(parts) == 2 and parts[1].isdigit():
            s = parts[0]
    return s if s else None


def build_s2_to_arxiv_bridge(cache_dir=None):
    """Load the semantic_scholar config and build {S2 paperId -> arXivId}."""
    print("Loading SciEvo semantic_scholar config (bridge table)...")
    from datasets import load_dataset
    s2 = load_dataset("Ahren09/SciEvo", "semantic_scholar", split="train",
                      cache_dir=cache_dir)
    bridge = {}
    for row in s2:
        pid = row.get("paperId")
        aid = row.get("arXivId")
        if pid and aid:
            clean = clean_arxiv_id(aid)
            if clean:
                bridge[pid] = clean
    print(f"  Bridge: {len(bridge):,} S2 paperIds mapped to arXiv IDs")
    return bridge


def build_edges(corpus_ids, bridge, cache_dir=None):
    """Load the references config and build internal citation edges.
    An edge (A, B) means paper A cites paper B, and both are in corpus_ids."""
    print("Loading SciEvo references config...")
    from datasets import load_dataset
    refs = load_dataset("Ahren09/SciEvo", "references", split="train",
                        cache_dir=cache_dir)

    corpus_set = set(corpus_ids)
    edges = []
    papers_with_refs = 0
    papers_with_internal_refs = 0

    print(f"  Building edges (corpus size: {len(corpus_set):,})...")
    for i, row in enumerate(refs):
        citing_id = clean_arxiv_id(row.get("arXivId"))
        if citing_id is None or citing_id not in corpus_set:
            continue

        ref_list = row.get("references") or []
        if not ref_list:
            continue
        papers_with_refs += 1

        found_internal = False
        for ref in ref_list:
            cited_paper = ref.get("citedPaper", {})
            cited_s2_id = cited_paper.get("paperId")
            if cited_s2_id is None:
                continue
            cited_arxiv = bridge.get(cited_s2_id)
            if cited_arxiv and cited_arxiv in corpus_set:
                edges.append((citing_id, cited_arxiv))
                found_internal = True

        if found_internal:
            papers_with_internal_refs += 1

        if (i + 1) % 500000 == 0:
            print(f"    processed {i+1:,} / {len(refs):,} reference rows...")

    print(f"  Done: {len(edges):,} internal edges, "
          f"{papers_with_internal_refs:,} papers with >=1 internal citation")
    return edges


def build_matrices(edges, data_dir, out_dir):
    """Build mat_cit[t][s] and bow_articles[t] for DSNTM training.
    
    mat_cit[t][s]: sparse binary [num_docs_at_t, num_docs_at_s] where
    entry (i, j) = 1 means doc i at time t cites doc j at time s.
    Only s < t (papers can only cite earlier papers).
    
    bow_articles[t]: dense normalized BOW [num_docs_at_t, V] — needed by
    DSNTM's citation loss to compute paper proportions phi.
    """
    import scipy.sparse as sp
    import torch

    print("Building time-sliced citation matrices...")

    with open(os.path.join(data_dir, "vocab.json")) as f:
        vocab = json.load(f)
    with open(os.path.join(data_dir, "dataset_meta.json")) as f:
        meta = json.load(f)
    V = len(vocab)
    T = meta["num_times"]
    years = meta["years"]

    # load train split to get the document ordering
    train_bow = sp.load_npz(os.path.join(data_dir, "train_bow.npz"))
    train_times = np.load(os.path.join(data_dir, "train_times.npy"))

    # We need arXiv IDs for the train docs — reload from pkl
    # The ordering must match train_bow (same indices from prepare_data.py)
    # Since prepare_data uses seed 2019, we replicate the split to get IDs
    print("  Replicating train split to recover document IDs...")
    with open(os.path.join(out_dir, "citation_graph.pkl"), "rb") as f:
        graph_data = pickle.load(f)

    # We need the raw pkl to get IDs in split order
    # Look for the pkl path from the graph data or ask user
    pkl_path = graph_data.get("pkl_path")
    if pkl_path is None or not os.path.exists(pkl_path):
        raise FileNotFoundError(
            "Cannot find the original pkl to recover train doc IDs. "
            "Re-run with --pkl pointing to scievo_cscl_cslg_2014_2021.pkl")

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    papers = raw["papers"].copy()
    papers = papers[(papers["year"] >= 2014) & (papers["year"] <= 2020)]
    papers = papers.reset_index(drop=True)

    rng = np.random.RandomState(2019)
    n = len(papers)
    perm = rng.permutation(n)
    n_train = int(round(0.85 * n))
    train_indices = perm[:n_train]

    # Get train doc IDs in the same order as train_bow
    train_ids = [papers["id"][i] for i in train_indices]
    train_years_raw = np.array([papers["year"][i] for i in train_indices])

    # Filter to docs that survived vocab filtering (keep = bow.sum > 1)
    # We need to match the exact filtering in prepare_data.py
    keep = np.asarray(train_bow.sum(axis=1)).ravel() > 0
    # Note: prepare_data uses > 1, but we check against actual train_bow size
    assert sum(keep) == train_bow.shape[0] or len(train_ids) == train_bow.shape[0], \
        f"Mismatch: {len(train_ids)} IDs vs {train_bow.shape[0]} BOW rows. " \
        "The filtering logic may differ — check prepare_data.py"

    # If lengths match without filtering, all docs survived
    if len(train_ids) != train_bow.shape[0]:
        train_ids = [tid for tid, k in zip(train_ids, keep) if k]
        train_years_raw = train_years_raw[keep]

    assert len(train_ids) == train_bow.shape[0], \
        f"After filtering: {len(train_ids)} IDs vs {train_bow.shape[0]} BOW rows"

    # Build per-slice doc lists
    doc_ids_per_slice = [[] for _ in range(T)]
    doc_local_idx = [[] for _ in range(T)]  # index into train_bow
    for i, (tid, t) in enumerate(zip(train_ids, train_times)):
        doc_ids_per_slice[t].append(tid)
        doc_local_idx[t].append(i)

    for t in range(T):
        print(f"  Slice {t} ({years[t]}): {len(doc_ids_per_slice[t])} docs")

    # Build edge lookup: citing_id -> set of cited_ids
    cite_lookup = defaultdict(set)
    for citing, cited in edges:
        cite_lookup[citing].add(cited)

    # Build mat_cit[t][s] for s < t
    mat_cit = [[None] * T for _ in range(T)]
    total_citations = 0
    for t in range(1, T):
        ids_t = doc_ids_per_slice[t]
        id2row_t = {aid: row for row, aid in enumerate(ids_t)}
        for s in range(t):
            ids_s = doc_ids_per_slice[s]
            id2col_s = {aid: col for col, aid in enumerate(ids_s)}
            rows, cols = [], []
            for row_idx, citing_id in enumerate(ids_t):
                cited_set = cite_lookup.get(citing_id, set())
                for col_idx, cited_id in enumerate(ids_s):
                    if cited_id in cited_set:
                        rows.append(row_idx)
                        cols.append(col_idx)
            mat = sp.csr_matrix(
                (np.ones(len(rows), dtype=np.float32),
                 (rows, cols)),
                shape=(len(ids_t), len(ids_s))
            )
            mat_cit[t][s] = mat
            total_citations += len(rows)
            if len(rows) > 0:
                print(f"    mat_cit[{t}][{s}]: {len(rows)} citations")

    print(f"  Total internal citations in training set: {total_citations:,}")

    # Build bow_articles[t]: dense normalized BOW per slice
    bow_articles = []
    for t in range(T):
        idx = doc_local_idx[t]
        bow_t = train_bow[idx].toarray().astype(np.float32)
        sums = bow_t.sum(axis=1, keepdims=True)
        sums = np.maximum(sums, 1e-10)
        bow_t = bow_t / sums
        bow_articles.append(torch.from_numpy(bow_t))

    result = {
        "mat_cit": mat_cit,
        "bow_articles": bow_articles,
        "doc_ids_per_slice": doc_ids_per_slice,
        "total_citations": total_citations,
    }
    out_path = os.path.join(out_dir, "citation_matrices.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  Saved {out_path}")
    return result


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # Load corpus IDs
    print(f"Loading corpus from {args.pkl}...")
    with open(args.pkl, "rb") as f:
        raw = pickle.load(f)
    corpus_ids = raw["papers"]["id"].tolist()
    print(f"  Corpus: {len(corpus_ids):,} papers")

    # Build the bridge and edges
    bridge = build_s2_to_arxiv_bridge(cache_dir=args.cache_dir)
    edges = build_edges(corpus_ids, bridge, cache_dir=args.cache_dir)

    # Save the raw edge list
    coverage = len(set(e[0] for e in edges)) / len(corpus_ids) if corpus_ids else 0
    graph_data = {
        "edges": edges,
        "num_edges": len(edges),
        "num_papers_with_citations": len(set(e[0] for e in edges)),
        "coverage": round(coverage, 4),
        "pkl_path": os.path.abspath(args.pkl),
    }
    out_path = os.path.join(args.out_dir, "citation_graph.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(graph_data, f)
    print(f"\nSaved {out_path}")
    print(f"  {len(edges):,} internal edges, "
          f"coverage = {coverage:.1%} of corpus papers have >=1 internal citation")

    # Optionally build the time-sliced matrices
    if args.build_matrices:
        build_matrices(edges, args.data_dir, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", type=str, default="scievo_cscl_cslg_2014_2021.pkl",
                   help="path to the raw corpus pkl")
    p.add_argument("--out_dir", type=str, default="data/citations")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="HuggingFace cache dir (if datasets already downloaded)")
    p.add_argument("--build_matrices", action="store_true",
                   help="also build mat_cit[t][s] for DSNTM training")
    p.add_argument("--data_dir", type=str, default="data/processed",
                   help="path to processed data (for --build_matrices)")
    main(p.parse_args())