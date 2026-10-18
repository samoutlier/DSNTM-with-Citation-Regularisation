"""
train.py
========
Trains DSNTM with early stopping on validation perplexity, optionally with
citation regularization (paper Sec. 5).

Optimization (defaults follow the paper's Appendix A / official code):
    Adam, lr 6e-4, weight decay 1.2e-6, batch size 512, bow normalization,
    delta = sigma^2 = 0.005, gamma^2 = 1, frozen 300-d skip-gram rho.

LR schedule: the official D-ETM/DSNTM "nonmono" annealing, adapted to
evaluate-every-N-epochs. Validation perplexity is computed every
--eval_every epochs. If the newest val ppl is worse than the best val ppl
seen at least --nonmono evaluations ago, lr is divided by --lr_factor (4.0),
floored at 1e-5.

Early stopping: training stops when val ppl has not improved by at least
--min_delta over the best value for --patience consecutive evaluations
(default patience=6 evaluations = 30 epochs, min_delta=1.0 ppl). The best
checkpoint (lowest val ppl) is what gets saved.

CITATION REGULARIZATION (--citation)
------------------------------------
Adds L_cit (Eq. 36) to the objective: a BCE between attention-propagated
citation predictions and the observed citation matrix. Requires
citation_matrices.pkl from build_citation_graph.py --build_matrices.

The loss is O(D^2) in the number of documents (paper Appendix A). At full
scale on this corpus the label matrices alone need ~6.7 GB, which does not
fit alongside the model on an 11 GB card. Documents are therefore
subsampled per time slice to the most-connected papers, following the
paper's own precedent of using the top-cited subset for its CS dataset.
Control this with --citation_docs_per_slice (default 4000, ~3 GB).

All console output is mirrored to <save_dir>/train.log; the validation
history is written to <save_dir>/val_history.json.

DOCUMENT-COMPLETION ROBUSTNESS (--bow_word_dropout)
----------------------------------------------------
Observed on this corpus: citation regularization improves full-document
perplexity, coherence, and diversity, but WORSENS document-completion
perplexity (theta inferred from half the words). Mechanism: get_citation_loss
always infers theta from the FULL bow_articles[t] (never split), and the
citation BCE term rewards theta for being sharply calibrated -- this leaks
into the SHARED q_theta/mu_q_theta network and biases theta-inference
towards expecting full evidence, even for documents unrelated to citation
regularization.

--bow_word_dropout p (0 < p < 1) counters this directly: at each training
step, the bag-of-words fed into the theta ENCODER has each vocabulary count
independently zeroed with probability p (torch.rand_like(bows) > p), while
the bag-of-words used as the RECONSTRUCTION TARGET (nll) is left at its
original, uncorrupted counts. This trains theta to be inferred correctly
from partial evidence while still being scored on reconstructing the full
document -- the same task structure as document-completion evaluation, so
training under it should directly transfer to better completion PPL. This
only touches the ordinary reconstruction pathway (train.py's batch loop);
get_citation_loss's use of the full bow_articles is untouched, since
citation prediction genuinely benefits from precise, full-evidence theta.
Default 0.0 (off, exact prior behavior, no extra randomness consumed).
Recommended: 0.3-0.5 when combined with --citation.

Usage
-----
    # without citation regularization (Phase 1 baseline)
    python train.py --data_dir data/processed --save_dir runs/dsntm_k30 \
        --num_topics 30 --epochs 400 --eval_every 5

    # with citation regularization
    python train.py --data_dir data/processed --save_dir runs/dsntm_k30_cit \
        --num_topics 30 --epochs 400 --eval_every 5 \
        --citation --citation_file data/citations/citation_matrices.pkl

    # with citation regularization + completion-robustness training
    python train.py --data_dir data/processed --save_dir runs/dsntm_k30_cit_v2 \
        --num_topics 30 --epochs 400 --eval_every 5 \
        --citation --citation_file data/citations/citation_matrices.pkl \
        --bow_word_dropout 0.4
"""
import argparse
import copy
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch
from torch import optim

from dsntm.model import DSNTM
from dsntm.data import load_dataset, get_rnn_input
from dsntm.evaluation import full_document_perplexity


def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("dsntm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(save_dir, "train.log"))):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# ---------------------------------------------------------------------------
# citation data loading + subsampling
# ---------------------------------------------------------------------------
def _slice_degrees(mat_cit, num_times, slice_sizes):
    """Total internal degree (citations made + citations received) of every
    document, per time slice. Used to pick the most-connected subset."""
    deg = [np.zeros(n, dtype=np.int64) for n in slice_sizes]
    for t in range(num_times):
        for s in range(num_times):
            m = mat_cit[t][s] if mat_cit[t] is not None else None
            if m is None:
                continue
            deg[t] += np.asarray(m.sum(axis=1)).ravel().astype(np.int64)  # out
            deg[s] += np.asarray(m.sum(axis=0)).ravel().astype(np.int64)  # in
    return deg


def load_citation_data(path, num_times, device, docs_per_slice, log):
    """Load citation_matrices.pkl, subsample to the most-connected documents
    per slice, and return (mat_cit, bow_articles) as dense torch tensors on
    `device`, ready for DSNTM.update_citation_data().

    Subsampling keeps the top --citation_docs_per_slice documents by total
    internal degree in each slice, then re-indexes every mat_cit[t][s]
    accordingly so rows and columns stay aligned with bow_articles.
    """
    log.info(f"loading citation data from {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    mat_cit_raw = data["mat_cit"]
    bow_raw = data["bow_articles"]

    slice_sizes = [b.shape[0] for b in bow_raw]
    log.info(f"  slice sizes (full): {slice_sizes}")
    total_cit = sum(
        int(mat_cit_raw[t][s].nnz)
        for t in range(num_times) for s in range(num_times)
        if mat_cit_raw[t] is not None and mat_cit_raw[t][s] is not None
    )
    log.info(f"  citations (full): {total_cit:,}")

    # ---- pick the most-connected documents in each slice ----------------
    deg = _slice_degrees(mat_cit_raw, num_times, slice_sizes)
    keep_idx = []
    for t in range(num_times):
        n = slice_sizes[t]
        if n <= docs_per_slice:
            idx = np.arange(n)
        else:
            idx = np.argpartition(deg[t], -docs_per_slice)[-docs_per_slice:]
            idx = np.sort(idx)
        keep_idx.append(idx)
    log.info(f"  slice sizes (subsampled to <= {docs_per_slice}): "
             f"{[len(i) for i in keep_idx]}")

    # ---- re-index the matrices and densify ------------------------------
    mat_cit = [[None] * num_times for _ in range(num_times)]
    kept_cit = 0
    for t in range(1, num_times):
        for s in range(t):
            m = mat_cit_raw[t][s]
            if m is None:
                continue
            sub = m[keep_idx[t]][:, keep_idx[s]]
            kept_cit += int(sub.nnz)
            mat_cit[t][s] = torch.from_numpy(
                sub.toarray().astype(np.float32)).to(device)
    log.info(f"  citations retained after subsampling: {kept_cit:,} "
             f"({kept_cit / max(total_cit, 1):.1%} of full graph)")

    bow_articles = [bow_raw[t][keep_idx[t]].to(device).float()
                    for t in range(num_times)]

    mem = (sum(b.numel() for b in bow_articles)
           + sum(mat_cit[t][s].numel()
                 for t in range(num_times) for s in range(num_times)
                 if mat_cit[t][s] is not None)) * 4 / 1e9
    log.info(f"  citation tensors on {device}: ~{mem:.2f} GB")
    return mat_cit, bow_articles


def main(args):
    log = setup_logging(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    vocab, meta, splits, embeddings = load_dataset(args.data_dir, device)
    if embeddings is None:
        raise FileNotFoundError(
            "embeddings_300d.npy not found -- run train_embeddings.py first")
    V, T = len(vocab), meta["num_times"]
    train, valid = splits["train"], splits["valid"]
    log.info(f"device={device}  V={V}  T={T}  "
             f"train={train.num_docs}  valid={valid.num_docs}")
    log.info(f"args: {vars(args)}")

    train_rnn_inp = get_rnn_input(train, T)
    valid_rnn_inp = get_rnn_input(valid, T)

    model = DSNTM(
        num_topics=args.num_topics, num_times=T, vocab_size=V,
        embeddings=embeddings, rho_size=args.rho_size,
        t_hidden_size=args.t_hidden_size,
        eta_hidden_size=args.eta_hidden_size, eta_nlayers=args.eta_nlayers,
        enc_drop=args.enc_drop, eta_dropout=args.eta_dropout,
        delta=args.delta, num_heads=args.num_heads,
        train_embeddings=bool(args.train_embeddings),
        citation=args.citation, c_weight=args.c_weight,
        citation_min_cites=args.citation_min_cites, device=device,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"learnable parameters: {n_params:,}")

    # ---- citation regularization ---------------------------------------
    if args.citation:
        if not os.path.exists(args.citation_file):
            raise FileNotFoundError(
                f"{args.citation_file} not found. Build it first:\n"
                "  python build_citation_graph.py --pkl <corpus.pkl> "
                "--out_dir data/citations --build_matrices "
                "--data_dir data/processed")
        mat_cit, bow_articles = load_citation_data(
            args.citation_file, T, device, args.citation_docs_per_slice, log)
        model.update_citation_data(mat_cit, bow_articles)
        log.info(f"citation regularization ON  (c_weight={args.c_weight}, "
                 f"min_citation_count={args.citation_min_cites})")
    else:
        log.info("citation regularization OFF")

    if args.bow_word_dropout > 0 and args.clip <= 0:
        log.info(
            "WARNING: --bow_word_dropout is on with --clip 0 (no gradient "
            "clipping). A rare fully-emptied document is safeguarded "
            "against, but unclipped gradients from ordinary variance in a "
            "batch can still occasionally spike and destabilize training. "
            "Recommended: rerun with --clip 2.0 (or similar) for safety.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.wdecay)

    # forward() recomputes the citation loss over the WHOLE citation subset
    # on every minibatch call; scale it by 1/num_batches so its cumulative
    # contribution across one epoch approximates being added once (Eq. 35
    # adds L_cit once per corpus pass, not once per minibatch).
    num_batches = max(1, (train.num_docs + args.batch_size - 1)
                      // args.batch_size)
    citation_scale = (1.0 / num_batches) if args.citation else 1.0
    if args.citation:
        log.info(f"citation_scale = 1/{num_batches} = {citation_scale:.5f}  "
                 f"(so L_cit contributes once per epoch, not once per batch)")

    best_ppl, best_state, best_epoch = float("inf"), None, 0
    val_history = []
    bad_evals = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(train.num_docs).numpy()
        acc = {"nelbo": 0, "nll": 0, "kl_theta": 0, "kl_eta": 0,
               "kl_alpha": 0, "cit": 0}
        cnt = 0
        for start in range(0, train.num_docs, args.batch_size):
            idx = perm[start:start + args.batch_size]
            bows, times = train.batch(idx)
            # bows stays the RECONSTRUCTION TARGET (nll scores against this,
            # full counts, untouched). bows_for_theta is the ENCODER INPUT
            # only -- optionally corrupted so theta learns to work from
            # partial evidence, directly targeting document-completion PPL.
            if args.bow_word_dropout > 0:
                keep_mask = (torch.rand_like(bows)
                            > args.bow_word_dropout).float()
                bows_for_theta = bows * keep_mask
                # safety: for a short document, chance can zero EVERY word,
                # leaving the encoder a fully-empty, degenerate input. If
                # that document's true bow (reconstruction target) is not
                # trivial, the resulting prediction error gets amplified by
                # coeff=num_docs/batch_size (~130x) into the batch loss --
                # exactly the kind of single-step spike that can destabilize
                # training when unclipped. Fall back to the untouched bow
                # for any row that dropout emptied completely.
                empty_rows = ((bows_for_theta.sum(dim=1) == 0)
                             & (bows.sum(dim=1) > 0))
                if empty_rows.any():
                    bows_for_theta = torch.where(
                        empty_rows.unsqueeze(1), bows, bows_for_theta)
            else:
                bows_for_theta = bows
            sums = bows_for_theta.sum(1, keepdim=True).clamp(min=1.0)
            nbows = bows_for_theta / sums if args.bow_norm else bows_for_theta
            optimizer.zero_grad()
            nelbo, nll, kl_eta, kl_theta, kl_alpha, c_loss = model(
                bows, nbows, times, train_rnn_inp, train.num_docs,
                citation_scale=citation_scale)
            nelbo.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            for k, v in zip(acc, (nelbo, nll, kl_theta, kl_eta, kl_alpha,
                                  c_loss)):
                acc[k] += v.item()
            cnt += 1

        lr_now = optimizer.param_groups[0]["lr"]
        msg = (f"epoch {epoch:3d}  lr {lr_now:.2e}  "
               f"nelbo {acc['nelbo']/cnt:.1f}  nll {acc['nll']/cnt:.1f}  "
               f"kl_theta {acc['kl_theta']/cnt:.1f}  "
               f"kl_eta {acc['kl_eta']/cnt:.2f}  "
               f"kl_alpha {acc['kl_alpha']/cnt:.1f}")
        if args.citation:
            msg += f"  cit {acc['cit']/cnt:.1f}"
        log.info(msg)

        # ---- validation every eval_every epochs ----------------------
        if epoch % args.eval_every != 0 and epoch != args.epochs:
            continue
        val_ppl = full_document_perplexity(model, valid, valid_rnn_inp,
                                           args.eval_batch_size)
        val_history.append({"epoch": epoch, "val_ppl": val_ppl,
                            "lr": lr_now})
        improved = val_ppl < best_ppl - args.min_delta
        log.info(f"  >> VAL PPL {val_ppl:.1f}  "
                 f"(best {best_ppl:.1f} @ {best_epoch})"
                 f"{'  *improved*' if improved else ''}")
        if val_ppl < best_ppl:
            best_ppl, best_epoch = val_ppl, epoch
            best_state = copy.deepcopy(model.state_dict())
        bad_evals = 0 if improved else bad_evals + 1

        # nonmono lr annealing over evaluation points
        ppls = [h["val_ppl"] for h in val_history]
        if (args.anneal_lr and len(ppls) > args.nonmono
                and val_ppl > min(ppls[:-args.nonmono]) and lr_now > 1e-5):
            optimizer.param_groups[0]["lr"] = lr_now / args.lr_factor
            log.info(f"  >> annealing lr -> "
                     f"{optimizer.param_groups[0]['lr']:.2e}")

        if bad_evals >= args.patience:
            log.info(f"early stopping: no val improvement > {args.min_delta} "
                     f"for {args.patience} evaluations")
            break

    elapsed = time.time() - start_time
    log.info(f"training done in {elapsed/60:.1f} min; "
             f"best val ppl {best_ppl:.1f} at epoch {best_epoch}")

    model.load_state_dict(best_state)
    ckpt = os.path.join(args.save_dir, "best_model.pt")
    torch.save({"state_dict": best_state, "args": vars(args),
                "best_val_ppl": best_ppl, "best_epoch": best_epoch}, ckpt)
    with open(os.path.join(args.save_dir, "val_history.json"), "w") as f:
        json.dump(val_history, f, indent=2)
    log.info(f"saved {ckpt}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/processed")
    p.add_argument("--save_dir", type=str, default="runs/dsntm")
    p.add_argument("--num_topics", type=int, default=30)
    p.add_argument("--rho_size", type=int, default=300)
    p.add_argument("--t_hidden_size", type=int, default=800)
    p.add_argument("--eta_hidden_size", type=int, default=200)
    p.add_argument("--eta_nlayers", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=10)
    p.add_argument("--delta", type=float, default=0.005)
    p.add_argument("--train_embeddings", type=int, default=0)
    p.add_argument("--enc_drop", type=float, default=0.0)
    p.add_argument("--eta_dropout", type=float, default=0.0)
    # citation regularization
    p.add_argument("--citation", action="store_true",
                   help="enable citation regularization (paper Sec. 5)")
    p.add_argument("--citation_file", type=str,
                   default="data/citations/citation_matrices.pkl")
    p.add_argument("--c_weight", type=float, default=1.0,
                   help="weight on the citation BCE term")
    p.add_argument("--citation_docs_per_slice", type=int, default=4000,
                   help="cap on documents per time slice used in the "
                        "citation loss (bounds the O(D^2) memory)")
    p.add_argument("--citation_min_cites", type=int, default=4,
                   help="a citing paper contributes to the loss only if it "
                        "has at least this many observed citations")
    # optimization
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--wdecay", type=float, default=1.2e-6)
    p.add_argument("--lr_factor", type=float, default=4.0)
    p.add_argument("--anneal_lr", type=int, default=1)
    p.add_argument("--nonmono", type=int, default=3,
                   help="evaluations (not epochs) of tolerance before anneal")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--patience", type=int, default=6,
                   help="early-stop after this many non-improving evals")
    p.add_argument("--min_delta", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=1000)
    p.add_argument("--bow_norm", type=int, default=1)
    p.add_argument("--bow_word_dropout", type=float, default=0.0,
                   help="probability of zeroing each vocab count in the "
                        "theta-ENCODER input only (reconstruction target "
                        "unaffected). Trains theta to work from partial "
                        "evidence -- targets document-completion PPL. "
                        "0.0 = off (exact prior behavior). Try 0.3-0.5.")
    p.add_argument("--clip", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=2019)
    main(p.parse_args())