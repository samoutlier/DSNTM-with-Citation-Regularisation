"""
dsntm/model.py  
Architecture :

  Topic embeddings alpha :
    - learned initial transformed embedding  mu_0_alpha in R^{1 x K x L}
      (LayerNorm-ed) plays the role of alpha~^(1) ~ N(0, sigma^2 I) 
    - at each step t: fused Q/K/V projection (single Linear L -> 3L) of the
      transformed embedding at t-1; keys/values are ACCUMULATED over all
      previous steps -> multi-head attention
    - residual connection + LayerNorm (Eq. 19-20); one LayerNorm module is
      shared for the initial normalization and all residual steps, exactly
      as in the official code
    - linear heads f_mu, f_sigma map the transformed embedding to
      q(alpha_k^(t)) = N(f_mu, f_sigma) (Eq. 14)
    - prior: alpha^(t) ~ N(alpha^(t-1)_sampled, delta * I); at t=0 the KL is
      taken against N(0, I) (official-code behavior; the paper's Eq. 21 uses
      sigma^2 -- documented deviation inherited from the official code)

  Topic proportion mean eta (identical to D-ETM):
    - linear map V -> 200, 3-layer LSTM(200), heads on [h_t ; eta_{t-1}]
    - prior N(eta_{t-1}, delta * I), N(0, I) at t=0

  Topic proportions theta (identical to D-ETM):
    - MLP([bow_norm ; eta_td]) with 800 hidden units, ReLU
      (official DSNTM uses TWO hidden layers of 800 -- kept)
    - prior LN(eta_td, I)

  beta^(t) = softmax(rho^T alpha^(t))  with frozen 300-d skip-gram rho.

  Citation regularization (Sec. 5, Eq. 28-36): implemented faithfully
  (theta -> attention -> paper proportion phi -> BCE against the citation
  matrix), gated by `citation=True` + `update_citation_data(...)`; OFF by
  default because no citation graph is available yet.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

TINY = 1e-10


class DSNTM(nn.Module):
    def __init__(self, num_topics, num_times, vocab_size, embeddings,
                 rho_size=300, t_hidden_size=800, eta_hidden_size=200,
                 eta_nlayers=3, enc_drop=0.0, eta_dropout=0.0,
                 delta=0.005, num_heads=10, train_embeddings=False,
                 citation=False, c_weight=1.0, citation_min_cites=4,
                 device="cpu"):
        super().__init__()
        self.num_topics = num_topics
        self.num_times = num_times
        self.vocab_size = vocab_size
        self.rho_size = rho_size
        self.eta_hidden_size = eta_hidden_size
        self.eta_nlayers = eta_nlayers
        self.enc_drop = enc_drop
        self.t_drop = nn.Dropout(enc_drop)
        self.delta = delta
        self.citation = citation
        self.c_weight = c_weight
        self.citation_min_cites = citation_min_cites
        self.device = device

        # ---- alpha / self-attention ----------------------------------
        self.mu_0_alpha = nn.Parameter(torch.randn(1, num_topics, rho_size))
        self.mu_q_alpha = nn.Linear(rho_size, rho_size, bias=True)
        self.logsigma_q_alpha = nn.Linear(rho_size, rho_size, bias=True)
        self.num_heads = num_heads
        self.multi_attn = nn.MultiheadAttention(
            embed_dim=rho_size, num_heads=num_heads, batch_first=True)
        self.qkv_proj = nn.Linear(rho_size, 3 * rho_size)
        self.layer_norm = nn.LayerNorm(rho_size)

        # ---- word embeddings rho -------------------------------------
        self.train_embeddings = train_embeddings
        if train_embeddings:
            self.rho = nn.Linear(rho_size, vocab_size, bias=False)
        else:
            self.register_buffer(
                "rho", embeddings.clone().float())  # [V, L], frozen

        # ---- theta encoder (D-ETM) -----------------------------------
        self.q_theta = nn.Sequential(
            nn.Linear(vocab_size + num_topics, t_hidden_size),
            nn.ReLU(inplace=False),
            nn.Linear(t_hidden_size, t_hidden_size),
            nn.ReLU(inplace=False),
        )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

        # ---- eta encoder (D-ETM LSTM) --------------------------------
        self.q_eta_map = nn.Linear(vocab_size, eta_hidden_size)
        self.q_eta = nn.LSTM(eta_hidden_size, eta_hidden_size, eta_nlayers,
                             dropout=eta_dropout)
        self.mu_q_eta = nn.Linear(eta_hidden_size + num_topics, num_topics)
        self.logsigma_q_eta = nn.Linear(eta_hidden_size + num_topics, num_topics)

        # ---- citation regularization state ---------------------------
        self.citation_loss_fn = nn.BCELoss(reduction="sum")
        self.mat_cit = None          # mat_cit[t][s]: docs at t x docs at s
        self.bow_articles = None     # bow_articles[t]: dense bow of docs at t
        self.num_cum_articles = [0] * (num_times + 1)
        self.label_matrices = [None] * num_times

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------
    def reparameterize(self, mu, logvar):
        eps = torch.randn(mu.shape, device=mu.device)
        return mu + eps * torch.exp(0.5 * logvar)

    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """KL( N(q_mu, exp(q_logsigma)) || N(p_mu, exp(p_logsigma)) ).
        Matches the official code, including the +1e-6 in the denominator."""
        if p_mu is not None and p_logsigma is not None:
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = (sigma_q_sq + (q_mu - p_mu) ** 2) / (sigma_p_sq + 1e-6)
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1)
        else:
            kl = -0.5 * torch.sum(
                1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1)
        return kl

    def init_hidden(self):
        w = next(self.parameters())
        return (w.new_zeros(self.eta_nlayers, 1, self.eta_hidden_size),
                w.new_zeros(self.eta_nlayers, 1, self.eta_hidden_size))

    # ------------------------------------------------------------------
    # alpha: self-attention over previous transformed topic embeddings
    # ------------------------------------------------------------------
    def get_attention(self, predict=False):
        """Returns (transformed_alphas [T(+1), K, L], attentions dict).
        attentions[t] is [K, K*(t)]-ish -- weights of topic k at time t over
        all K*(t) previous (time, topic) slots (official code squeezes the
        batch dim). predict=True runs one extra step -> T+1 extrapolation."""
        alpha = self.mu_0_alpha
        num_times = self.num_times + 1 if predict else self.num_times
        alphas = torch.empty((num_times, self.num_topics, self.rho_size),
                             device=alpha.device)
        alpha = self.layer_norm(alpha)
        alphas[0] = torch.squeeze(alpha, 0)
        attentions = {}
        cul_k = cul_v = None
        for t in range(1, num_times):
            qkv = self.qkv_proj(alpha)
            q_t, k_t, v_t = qkv.chunk(3, dim=-1)
            prev_alpha = alpha
            if t == 1:
                cul_k, cul_v = k_t, v_t
            else:
                cul_k = torch.cat((cul_k, k_t), 1)
                cul_v = torch.cat((cul_v, v_t), 1)
            alpha, attn_weights = self.multi_attn(q_t, cul_k, cul_v)
            attentions[t] = torch.squeeze(attn_weights, 0)
            alpha = self.layer_norm(alpha + prev_alpha)  # residual + LN
            alphas[t] = torch.squeeze(alpha, 0)
        return alphas, attentions

    def get_alpha(self):
        """Training-time alpha: sample q(alpha^(t)) and accumulate KL."""
        outputs, attentions = self.get_attention()
        alphas = torch.empty(
            (self.num_times, self.num_topics, self.rho_size),
            device=outputs.device)
        kl_alpha = []

        q_mu_0 = self.mu_q_alpha(outputs[0])
        q_logsigma_0 = self.logsigma_q_alpha(outputs[0])
        alphas[0] = self.reparameterize(q_mu_0, q_logsigma_0)
        p_mu_0 = torch.zeros_like(q_mu_0)
        p_logsigma_0 = torch.zeros_like(q_logsigma_0)
        kl_alpha.append(self.get_kl(q_mu_0, q_logsigma_0, p_mu_0, p_logsigma_0))

        for t in range(1, self.num_times):
            q_mu_t = self.mu_q_alpha(outputs[t])
            q_logsigma_t = self.logsigma_q_alpha(outputs[t])
            alphas[t] = self.reparameterize(q_mu_t, q_logsigma_t)
            p_mu_t = alphas[t - 1]
            p_logsigma_t = torch.log(
                self.delta * torch.ones_like(q_logsigma_t))
            kl_alpha.append(self.get_kl(q_mu_t, q_logsigma_t, p_mu_t, p_logsigma_t))

        return alphas, attentions, torch.stack(kl_alpha).sum()

    @torch.no_grad()
    def get_alpha_mean(self, predict=False):
        """Deterministic alpha (posterior means) for evaluation; with
        predict=True also returns the extrapolated alpha^(T+1)."""
        outputs, attentions = self.get_attention(predict)
        return self.mu_q_alpha(outputs), attentions

    # ------------------------------------------------------------------
    # eta (D-ETM)
    # ------------------------------------------------------------------
    def get_eta(self, rnn_inp):
        inp = self.q_eta_map(rnn_inp).unsqueeze(1)   # [T, 1, H]
        output, _ = self.q_eta(inp, self.init_hidden())
        output = output.squeeze(1)                   # [T, H]

        etas = torch.zeros(self.num_times, self.num_topics,
                           device=rnn_inp.device)
        kl_eta = []
        zeros_k = torch.zeros(self.num_topics, device=rnn_inp.device)
        inp_0 = torch.cat([output[0], zeros_k], dim=0)
        mu_0 = self.mu_q_eta(inp_0)
        logsigma_0 = self.logsigma_q_eta(inp_0)
        etas[0] = self.reparameterize(mu_0, logsigma_0)
        kl_eta.append(self.get_kl(mu_0, logsigma_0, zeros_k,
                                  torch.zeros_like(zeros_k)))
        for t in range(1, self.num_times):
            inp_t = torch.cat([output[t], etas[t - 1]], dim=0)
            mu_t = self.mu_q_eta(inp_t)
            logsigma_t = self.logsigma_q_eta(inp_t)
            etas[t] = self.reparameterize(mu_t, logsigma_t)
            p_logsigma_t = torch.log(self.delta * torch.ones_like(logsigma_t))
            kl_eta.append(self.get_kl(mu_t, logsigma_t, etas[t - 1],
                                      p_logsigma_t))
        return etas, torch.stack(kl_eta).sum()

    @torch.no_grad()
    def get_eta_mean(self, rnn_inp):
        inp = self.q_eta_map(rnn_inp).unsqueeze(1)
        output, _ = self.q_eta(inp, self.init_hidden())
        output = output.squeeze(1)
        etas = torch.zeros(self.num_times, self.num_topics,
                           device=rnn_inp.device)
        zeros_k = torch.zeros(self.num_topics, device=rnn_inp.device)
        etas[0] = self.mu_q_eta(torch.cat([output[0], zeros_k], dim=0))
        for t in range(1, self.num_times):
            etas[t] = self.mu_q_eta(torch.cat([output[t], etas[t - 1]], dim=0))
        return etas

    # ------------------------------------------------------------------
    # theta (D-ETM)
    # ------------------------------------------------------------------
    def get_theta(self, eta, normalized_bows, times=None):
        if times is not None:
            eta_td = eta[times.long()]
        else:
            eta_td = eta.repeat((normalized_bows.shape[0], 1)) \
                if eta.dim() == 1 else eta
        inp = torch.cat([normalized_bows, eta_td], dim=1)
        q_theta = self.q_theta(inp)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        kl_theta = self.get_kl(
            mu_theta, logsigma_theta, eta_td,
            torch.zeros(self.num_topics, device=eta_td.device))
        return theta, kl_theta

    @torch.no_grad()
    def get_theta_mean(self, eta_td, normalized_bows):
        inp = torch.cat([normalized_bows, eta_td], dim=1)
        q_theta = self.q_theta(inp)
        return F.softmax(self.mu_q_theta(q_theta), dim=-1)

    # ------------------------------------------------------------------
    # beta
    # ------------------------------------------------------------------
    def get_beta(self, alpha):
        """alpha [..., K, L] -> beta [..., K, V]."""
        if self.train_embeddings:
            logit = self.rho(alpha.reshape(-1, self.rho_size))
        else:
            logit = torch.mm(alpha.reshape(-1, self.rho_size),
                             self.rho.t())
        logit = logit.reshape(*alpha.shape[:-1], self.vocab_size)
        return F.softmax(logit, dim=-1)

    def get_nll(self, theta, beta, bows):
        """theta [B,K], beta [B,K,V], bows [B,V] -> per-doc NLL [B]."""
        loglik = torch.bmm(theta.unsqueeze(1), beta).squeeze(1)
        loglik = torch.log(loglik + TINY)
        return -(loglik * bows).sum(-1)

    # ------------------------------------------------------------------
    # citation regularization (Sec. 5) -- flag-gated, OFF by default
    # ------------------------------------------------------------------
    def update_citation_data(self, mat_cit, bow_articles):
        """mat_cit[t][s]: {0,1} matrix [docs_at_t, docs_at_s] (s < t).
        bow_articles[t]: dense normalized bow of all docs at time t."""
        self.mat_cit = mat_cit
        self.bow_articles = bow_articles
        if mat_cit is None:
            self.num_cum_articles = [0] * (self.num_times + 1)
            self.label_matrices = [None] * self.num_times
            return
        for t in range(1, self.num_times):
            self.num_cum_articles[t] = (self.num_cum_articles[t - 1]
                                        + self.bow_articles[t - 1].shape[0])
            mat_label = torch.zeros(self.bow_articles[t].shape[0],
                                    self.num_cum_articles[t],
                                    device=self.bow_articles[t].device)
            for s in range(t):
                mat_label[:, self.num_cum_articles[s]:
                          self.num_cum_articles[s + 1]] = self.mat_cit[t][s]
            self.label_matrices[t] = mat_label

    def get_phi(self, theta):
        """Paper proportion phi (Eq. 32): column-normalized theta^T."""
        tmp = theta.t()
        return tmp / tmp.sum(dim=1, keepdim=True)

    def get_citation_loss(self, attentions, etas, min_citation_count=4):
        """Eq. 28-36: BCE between attention-propagated citation predictions
        and observed citation labels. Returns (loss, kl_theta_over_all_docs)
        -- the official code REPLACES the batch kl_theta by this full-corpus
        kl_theta when citation regularization is on; we keep that behavior."""
        kl_theta_total = 0.0
        thetas, phis_by_t = {}, {}
        for t in range(self.num_times):
            theta, kl = self.get_theta(etas[t], self.bow_articles[t])
            thetas[t] = theta
            phis_by_t[t] = self.get_phi(theta)
            kl_theta_total = kl_theta_total + kl.sum()

        total_docs = self.num_cum_articles[self.num_times - 1]
        phis = torch.zeros(self.num_times * self.num_topics, total_docs,
                           device=etas.device)
        for t in range(1, self.num_times):
            phis[(t - 1) * self.num_topics: t * self.num_topics,
                 self.num_cum_articles[t - 1]: self.num_cum_articles[t]] \
                = phis_by_t[t - 1]

        total_loss = 0.0
        for t in range(1, self.num_times):
            att = attentions[t]  # [K, K*t]
            cum_phi = phis[: t * self.num_topics,
                           : self.num_cum_articles[t]].detach()
            label = self.label_matrices[t]
            paper_to_prev_topics = torch.mm(thetas[t], att)
            preds = torch.mm(paper_to_prev_topics, cum_phi).clamp(0.0, 1.0)
            keep = label.sum(dim=1) >= min_citation_count
            if keep.any():
                total_loss = total_loss + self.citation_loss_fn(
                    preds[keep], label[keep])
        return total_loss, kl_theta_total

    # ------------------------------------------------------------------
    # forward: NELBO (Eq. 27 / 35)
    # ------------------------------------------------------------------
    def forward(self, bows, normalized_bows, times, rnn_inp, num_docs,
                citation_scale=1.0):
        bsz = normalized_bows.size(0)
        coeff = num_docs / bsz
        alpha, attentions, kl_alpha = self.get_alpha()
        eta, kl_eta = self.get_eta(rnn_inp)
        theta, kl_theta = self.get_theta(eta, normalized_bows, times)
        kl_theta = kl_theta.sum() * coeff
        beta = self.get_beta(alpha)          # [T, K, V]
        beta = beta[times.long()]            # [B, K, V]
        nll = self.get_nll(theta, beta, bows).sum() * coeff

        if self.citation and self.c_weight > 0 and self.mat_cit is not None:
            citation_loss, kl_theta_full = self.get_citation_loss(
                attentions, eta, self.citation_min_cites)
            # kl_theta_full and citation_loss are computed over the WHOLE
            # citation subset (all retained docs across all slices), not
            # just this minibatch. Eq. 35 adds L_cit once per pass over the
            # corpus; this forward() runs once per MINIBATCH (~num_batches
            # times per epoch), so without scaling, the citation term and
            # its replacement kl_theta get added num_batches times too many
            # per epoch, drowning out nll's properly-scaled contribution
            # and destabilizing training. Scaling by citation_scale
            # (1/num_batches, set in train.py) makes their cumulative
            # contribution across one epoch approximate being added once.
            kl_theta = kl_theta_full * citation_scale
            citation_loss = citation_loss * self.c_weight * citation_scale
        else:
            citation_loss = torch.tensor(0.0, device=bows.device)

        nelbo = nll + kl_eta + kl_theta + kl_alpha + citation_loss
        return nelbo, nll, kl_eta, kl_theta, kl_alpha, citation_loss

    # ------------------------------------------------------------------
    # T+1 prediction (paper Sec. 7.2)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_next(self, rnn_inp):
        """Extrapolates one step beyond the training window.
        Returns (alpha_pred [K, L], beta_pred [K, V], eta_pred [K]).
        alpha^(T+1) comes from running the shared self-attention one extra
        step (Eq. 6/13/14); eta^(T+1) uses the prior mean eta_T (since
        p(eta_{T+1} | eta_T) = N(eta_T, delta^2 I))."""
        alphas, _ = self.get_alpha_mean(predict=True)
        alpha_pred = alphas[self.num_times]
        beta_pred = self.get_beta(alpha_pred.unsqueeze(0)).squeeze(0)
        eta_pred = self.get_eta_mean(rnn_inp)[self.num_times - 1]
        return alpha_pred, beta_pred, eta_pred