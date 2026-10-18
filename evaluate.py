""" 
Loads the best checkpoint and computes/export everything:

  metrics.json
      val_ppl_full, test_ppl_full, test_ppl_completion,
      coherence_npmi (per slice + mean), diversity (per slice + mean),
      topic_quality (= mean coherence x mean diversity),
      holdout2021_ppl_full / holdout2021_ppl_completion_style
      (T+1 forecast validation: 2021 docs scored under the PREDICTED
      beta^(T+1), with eta^(T+1) = eta_T, its prior mean)
  beta_per_time.npy         [T, K, V]
  alpha_per_time.npy        [T, K, L]   (posterior means)
  beta_pred_2021.npy        [K, V]
  alpha_pred_2021.npy       [K, L]
  attention_weights.npz     att_t{t}: [K, K*t] for t = 1..T-1 (+ T for the
                            prediction step)
  top_words.txt / top_words.json    top-15 words per topic per slice (+2021
                            prediction)

Formulas are documented in dsntm/evaluation.py.

Usage
-----
    python evaluate.py --data_dir data/processed --save_dir runs/dsntm_k30
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from dsntm.model import DSNTM
from dsntm.data import load_dataset, get_rnn_input
from dsntm.evaluation import (NPMIScorer, coherence_diversity,
                              completion_perplexity,
                              full_document_perplexity, top_words, TINY)


@torch.no_grad()
def holdout_2021_perplexity(model, split, beta_pred, eta_pred,
                            batch_size=1000):
    """Full-document perplexity of the 2021 holdout under the predicted
    T+1 topics. theta is inferred with the trained encoder using
    eta^(T+1) := eta_T (the prior mean of p(eta_{T+1}|eta_T))."""
    acc, cnt = 0.0, 0
    for start in range(0, split.num_docs, batch_size):
        idx = np.arange(start, min(start + batch_size, split.num_docs))
        bows, _ = split.batch(idx)
        sums = bows.sum(1, keepdim=True).clamp(min=1.0)
        eta_td = eta_pred.unsqueeze(0).repeat(bows.shape[0], 1)
        theta = model.get_theta_mean(eta_td, bows / sums)
        loglik = torch.log(theta @ beta_pred + TINY)
        nll = -(loglik * bows).sum(-1)
        acc += (nll / sums.squeeze(1)).mean().item()
        cnt += 1
    return math.exp(acc / cnt)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, meta, splits, embeddings = load_dataset(args.data_dir, device)
    V, T = len(vocab), meta["num_times"]

    ckpt_path = os.path.join(args.save_dir, "best_model.pt")
    try:  # torch >= 1.13 has weights_only; default flips to True in 2.6
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    targs = ckpt["args"]
    model = DSNTM(
        num_topics=targs["num_topics"], num_times=T, vocab_size=V,
        embeddings=embeddings, rho_size=targs["rho_size"],
        t_hidden_size=targs["t_hidden_size"],
        eta_hidden_size=targs["eta_hidden_size"],
        eta_nlayers=targs["eta_nlayers"], delta=targs["delta"],
        num_heads=targs["num_heads"],
        train_embeddings=bool(targs["train_embeddings"]), device=device,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    train, valid, test = splits["train"], splits["valid"], splits["test"]
    train_rnn = get_rnn_input(train, T)
    valid_rnn = get_rnn_input(valid, T)
    test_rnn = get_rnn_input(test, T)

    metrics = {"best_epoch": ckpt["best_epoch"],
               "best_val_ppl_during_training": ckpt["best_val_ppl"]}

    print("Perplexities...")
    metrics["val_ppl_full"] = full_document_perplexity(model, valid, valid_rnn)
    metrics["test_ppl_full"] = full_document_perplexity(model, test, test_rnn)
    metrics["test_ppl_completion"] = completion_perplexity(
        model, splits["test_h1"], splits["test_h2"], test_rnn)
    print(f"  val(full)={metrics['val_ppl_full']:.1f}  "
          f"test(full)={metrics['test_ppl_full']:.1f}  "
          f"test(completion h1/h2)={metrics['test_ppl_completion']:.1f}")

    print("Exporting alpha/beta/attention...")
    alphas, attentions = model.get_alpha_mean(predict=True)   # [T+1, K, L]
    beta_all = model.get_beta(alphas)                         # [T+1, K, V]
    alpha_np = alphas.cpu().numpy()
    beta_np = beta_all.cpu().numpy()
    np.save(os.path.join(args.save_dir, "alpha_per_time.npy"), alpha_np[:T])
    np.save(os.path.join(args.save_dir, "beta_per_time.npy"), beta_np[:T])
    np.save(os.path.join(args.save_dir, "alpha_pred_2021.npy"), alpha_np[T])
    np.save(os.path.join(args.save_dir, "beta_pred_2021.npy"), beta_np[T])
    np.savez(os.path.join(args.save_dir, "attention_weights.npz"),
             **{f"att_t{t}": a.cpu().numpy() for t, a in attentions.items()})

    print("Coherence / diversity (reference corpus = train split, "
          "whole-document NPMI, top-10; diversity top-25)...")
    scorer = NPMIScorer(train.bow)
    coh_t, div_t = [], []
    for t in range(T):
        c, d = coherence_diversity(beta_np[t], scorer)
        coh_t.append(c)
        div_t.append(d)
        print(f"  slice {t} ({meta['years'][t]}): NPMI={c:.4f}  div={d:.3f}")
    metrics["coherence_npmi_per_slice"] = coh_t
    metrics["diversity_per_slice"] = div_t
    metrics["coherence_npmi"] = float(np.mean(coh_t))
    metrics["diversity"] = float(np.mean(div_t))
    metrics["topic_quality"] = metrics["coherence_npmi"] * metrics["diversity"]

    print("T+1 forecast validation on held-out 2021...")
    _, beta_pred, eta_pred = model.predict_next(train_rnn)
    metrics["holdout2021_ppl_full"] = holdout_2021_perplexity(
        model, splits["holdout2021"], beta_pred, eta_pred)
    # sanity baseline: score 2021 under the LAST TRAINED slice's beta
    metrics["holdout2021_ppl_using_last_slice_beta"] = \
        holdout_2021_perplexity(model, splits["holdout2021"],
                                beta_all[T - 1], eta_pred)
    print(f"  2021 ppl (predicted beta^(T+1)) = "
          f"{metrics['holdout2021_ppl_full']:.1f}  vs  "
          f"(frozen beta^(T)) = "
          f"{metrics['holdout2021_ppl_using_last_slice_beta']:.1f}")

    print("Top words...")
    tw = top_words(beta_np, vocab, num=10)   # T+1 slices incl. prediction
    labels = [str(y) for y in meta["years"]] + ["2021 (PREDICTED)"]
    with open(os.path.join(args.save_dir, "top_words.txt"), "w") as f:
        for t, label in enumerate(labels):
            f.write(f"===== time slice {t}: {label} =====\n")
            for k, words in enumerate(tw[t]):
                f.write(f"Topic {k:2d}: {' '.join(words)}\n")
            f.write("\n")
    with open(os.path.join(args.save_dir, "top_words.json"), "w") as f:
        json.dump({"labels": labels, "top_words": tw}, f, indent=2)

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Wrote metrics.json:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/processed")
    p.add_argument("--save_dir", type=str, default="runs/dsntm")
    main(p.parse_args())
