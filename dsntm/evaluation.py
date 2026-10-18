"""
dsntm/evaluation.py
===================
Metrics, with the exact formulas documented so numbers are comparable to
other D-ETM/DTM results.

Perplexity (two protocols)
--------------------------
1. FULL-DOCUMENT:
       ppl = exp( mean_over_batches( mean_d  nll_d / N_d ) )
   where nll_d = -sum_v bow_{d,v} log( theta_d^T beta^{(t_d)} )_v, with
   theta_d inferred from the FULL normalized bow and posterior means used
   everywhere (no sampling). Matches EvaluationModel.get_perplexity in the
   official code, including the mean-of-batch-means aggregation.
2. DOCUMENT COMPLETION (D-ETM protocol): theta_d inferred from the
   normalized FIRST half (h1); nll evaluated on the SECOND half (h2);
       ppl = exp( mean_over_batches( mean_d  nll_d(h2) / N_d(h2) ) ).

Coherence (NPMI)
----------------
Average NPMI over all 45 pairs of the TOP-10 words of each topic
    NPMI(wi,wj) = log[ P(wi,wj) / (P(wi)P(wj)) ] / ( -log P(wi,wj) ) 

Diversity
---------
Fraction of unique words among the top-25 words of all topics
(paper Eq. 40), per slice and averaged.

Topic quality = coherence x diversity (Dieng et al. convention).
"""
import math 
import numpy as np
import torch
 
# perplexity 
TINY = 1e-10


@torch.no_grad()
def full_document_perplexity(model, split, rnn_inp, batch_size=1000):
    model.eval()
    alphas, _ = model.get_alpha_mean()
    beta = model.get_beta(alphas)                    # [T, K, V]
    eta = model.get_eta_mean(rnn_inp)                # [T, K]
    acc, cnt = 0.0, 0
    for start in range(0, split.num_docs, batch_size):
        idx = np.arange(start, min(start + batch_size, split.num_docs))
        bows, times = split.batch(idx)
        sums = bows.sum(1, keepdim=True).clamp(min=1.0)
        theta = model.get_theta_mean(eta[times.long()], bows / sums)
        b = beta[times.long()]
        loglik = torch.log(torch.bmm(theta.unsqueeze(1), b).squeeze(1) + TINY)
        nll = -(loglik * bows).sum(-1)
        acc += (nll / sums.squeeze(1)).mean().item()
        cnt += 1
    return math.exp(acc / cnt)


@torch.no_grad()
def completion_perplexity(model, h1, h2, rnn_inp, batch_size=1000):
    """theta from h1, likelihood on h2 ."""
    model.eval()
    alphas, _ = model.get_alpha_mean()
    beta = model.get_beta(alphas)
    eta = model.get_eta_mean(rnn_inp)
    acc, cnt = 0.0, 0
    for start in range(0, h1.num_docs, batch_size):
        idx = np.arange(start, min(start + batch_size, h1.num_docs))
        bows1, times = h1.batch(idx)
        bows2, _ = h2.batch(idx)
        sums1 = bows1.sum(1, keepdim=True).clamp(min=1.0)
        sums2 = bows2.sum(1).clamp(min=1.0)
        theta = model.get_theta_mean(eta[times.long()], bows1 / sums1)
        b = beta[times.long()]
        loglik = torch.log(torch.bmm(theta.unsqueeze(1), b).squeeze(1) + TINY)
        nll = -(loglik * bows2).sum(-1)
        acc += (nll / sums2).mean().item()
        cnt += 1
    return math.exp(acc / cnt)

 
# coherence / diversity 
class NPMIScorer:
    """Fast NPMI against a reference corpus (list of token-id sets is
    replaced by a sparse binary doc-term matrix for speed)."""

    def __init__(self, ref_bow_csr):
        binary = ref_bow_csr.copy()
        binary.data = np.ones_like(binary.data)
        self.binary = binary.tocsc()
        self.num_docs = binary.shape[0]
        self.doc_freq = np.asarray(binary.sum(axis=0)).ravel()

    def topic_npmi(self, top_word_ids):
        n = len(top_word_ids)
        cols = self.binary[:, top_word_ids].toarray().astype(np.float64)
        co = cols.T @ cols                       # pairwise co-doc counts
        D = self.num_docs
        p_i = self.doc_freq[top_word_ids] / D
        vals = []
        for i in range(n):
            for j in range(i + 1, n):
                p_ij = co[i, j] / D
                if p_ij == 0:
                    vals.append(-1.0)            # standard NPMI convention
                    continue
                pmi = math.log(p_ij / (p_i[i] * p_i[j] + TINY) + TINY)
                vals.append(pmi / (-math.log(p_ij + TINY)))
        return float(np.mean(vals))


def coherence_diversity(beta_t, scorer, num_coh=10, num_div=25):
    """beta_t: [K, V] numpy. Returns (mean NPMI over topics, diversity)."""
    K = beta_t.shape[0]
    coh = []
    top_div = np.zeros((K, num_div), dtype=np.int64)
    for k in range(K):
        order = np.argsort(beta_t[k])[::-1]
        coh.append(scorer.topic_npmi(order[:num_coh]))
        top_div[k] = order[:num_div]
    diversity = len(np.unique(top_div)) / (K * num_div)
    return float(np.mean(coh)), diversity


def top_words(beta, vocab, num=15):
    """beta [T, K, V] -> nested list of top words per slice per topic."""
    out = []
    for t in range(beta.shape[0]):
        slice_words = []
        for k in range(beta.shape[1]):
            order = np.argsort(beta[t, k])[::-1][:num]
            slice_words.append([vocab[i] for i in order])
        out.append(slice_words)
    return out
