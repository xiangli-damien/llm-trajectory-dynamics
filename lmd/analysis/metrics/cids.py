import numpy as np
from typing import Dict, List, Optional, Tuple
from ..core.metrics_base import MetricBase
from ..core.types import MetricDirection, MetricOutput

def oas_shrinkage(S: np.ndarray, n_samples: int) -> float:
    d = S.shape[0]
    tr_S = np.trace(S)
    tr_S2 = float(np.sum(S * S))
    mu = tr_S / d
    num = (1.0 - 2.0/d) * tr_S2 + tr_S * tr_S
    den = (n_samples + 1.0 - 2.0/d) * (tr_S2 - (tr_S * tr_S) / d) + 1e-12
    alpha = num / den
    return float(np.clip(alpha, 0.0, 1.0))

class CIDS(MetricBase):
    def __init__(self, tail_window: int = 10, var_ratio: float = 0.95,
                 use_delta_last: bool = False, use_tail_variance: bool = False,
                 gamma: float = 1e-6, cv_folds: int = 5, seed: int = 42,
                 standardize: bool = False, max_oas_alpha: float = 0.15):
        super().__init__(tail_window=tail_window, var_ratio=var_ratio,
                         use_delta_last=use_delta_last, use_tail_variance=use_tail_variance,
                         gamma=gamma, cv_folds=cv_folds, seed=seed, 
                         standardize=standardize, max_oas_alpha=max_oas_alpha)
        self.tail_window = tail_window
        self.var_ratio = var_ratio
        self.use_delta_last = use_delta_last
        self.use_tail_variance = use_tail_variance
        self.gamma = gamma
        self.cv_folds = cv_folds
        self.seed = seed
        self.standardize = standardize
        self.max_oas_alpha = max_oas_alpha

    @property
    def name(self):
        return "cids"
    
    @property
    def requires_lm_head(self):
        return True
    
    @property
    def supported_modes(self):
        return ["state"]
    
    @property
    def output_specs(self):
        return {
            "cids_score": MetricDirection.HIGHER_BETTER,
            "cids_margin": MetricDirection.HIGHER_BETTER
        }
    
    def extract_features(self, ctx, states):
        N, L, H = states.shape
        k = min(self.tail_window, L)
        Q = ctx.get_shared_cache('Q')
        if Q is None:
            Q = ctx.lm_head.readout_projection(var_ratio=self.var_ratio)
            ctx.set_shared_cache('Q', Q)
        q = np.tensordot(states, Q, axes=([2],[0]))
        q_tail = q[:, -k:, :]
        y1 = q_tail.mean(axis=1).astype(np.float32)
        feats = [y1]
        if self.use_delta_last and L >= 2:
            delta_last = (q[:, -1, :] - q[:, -2, :]).astype(np.float32)
            feats.append(delta_last)
        if self.use_tail_variance and k >= 2:
            tail_var = q_tail.var(axis=1)
            feats.append(tail_var)
        Y = np.concatenate(feats, axis=1).astype(np.float32)
        return Y

    def fit_lda(self, Y: np.ndarray, labels: np.ndarray):
        y1 = Y[labels == 1]
        y0 = Y[labels == 0]
        d = Y.shape[1]
        if len(y1) == 0 or len(y0) == 0:
            return np.zeros(d, np.float32), np.zeros(d, np.float32)
        mu1, mu0 = y1.mean(0), y0.mean(0)
        n1, n0 = len(y1), len(y0)
        n = n1 + n0
        C1 = np.cov(y1.T, bias=False) if n1 > 1 else np.eye(d, dtype=np.float32)
        C0 = np.cov(y0.T, bias=False) if n0 > 1 else np.eye(d, dtype=np.float32)
        Sw = ((n1-1)*C1 + (n0-1)*C0) / max(1, (n-2))
        alpha = oas_shrinkage(Sw, n_samples=n)
        alpha = float(min(alpha, self.max_oas_alpha))
        mu = np.trace(Sw) / d
        Sigma = (1.0 - alpha) * Sw + alpha * (mu * np.eye(d, dtype=np.float32)) + self.gamma * np.eye(d, dtype=np.float32)
        try:
            w = np.linalg.solve(Sigma, (mu1 - mu0))
        except:
            w = mu1 - mu0
        m = 0.5 * (mu1 + mu0)
        return w.astype(np.float32), m.astype(np.float32)

    def score_samples(self, Y: np.ndarray, w: np.ndarray, m: np.ndarray):
        centered = Y - m[None, :]
        s = centered @ w
        margins = np.abs(s) / (np.linalg.norm(w) + 1e-12)
        return s.astype(np.float32), margins.astype(np.float32)

    def zscore_fit(self, X):
        mu = X.mean(0)
        sd = X.std(0) + 1e-12
        return mu.astype(np.float32), sd.astype(np.float32)

    def zscore_apply(self, X, mu, sd):
        return ((X - mu[None, :]) / sd[None, :]).astype(np.float32)

    def compute_state(self, ctx, states):
        Y_raw = self.extract_features(ctx, states)
        labels = ctx.labels.astype(int)
        N = len(labels)

        rng = np.random.RandomState(self.seed)
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            zeros = np.zeros(N, dtype=np.float32)
            return MetricOutput(
                name=self.name,
                scores={"cids_score": zeros, "cids_margin": zeros},
                directions=self.output_specs,
                cache_state={"cids_features": Y_raw, "cids_score": zeros, "cids_margin": zeros}
            )
        
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        folds = min(self.cv_folds, n_pos, n_neg)
        if folds < 2:
            Y = Y_raw
            if self.standardize:
                mu, sd = self.zscore_fit(Y)
                Y = self.zscore_apply(Y, mu, sd)
            w, m = self.fit_lda(Y, labels)
            s, g = self.score_samples(Y, w, m)
            return MetricOutput(
                name=self.name,
                scores={"cids_score": s, "cids_margin": g},
                directions=self.output_specs,
                cache_state={"cids_features": Y_raw, "cids_discriminant": {"w": w, "m": m},
                             "cids_score": s, "cids_margin": g}
            )

        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        pos_splits = np.array_split(pos_idx, folds)
        neg_splits = np.array_split(neg_idx, folds)
        s_all = np.zeros(N, dtype=np.float32)
        g_all = np.zeros(N, dtype=np.float32)

        for k in range(folds):
            val_idx = np.concatenate([pos_splits[k], neg_splits[k]])
            tr_idx = np.concatenate([np.concatenate([pos_splits[i] for i in range(folds) if i != k]),
                                     np.concatenate([neg_splits[i] for i in range(folds) if i != k])])
            Y_tr, Y_te = Y_raw[tr_idx], Y_raw[val_idx]
            y_tr = labels[tr_idx]

            if self.standardize:
                mu, sd = self.zscore_fit(Y_tr)
                Y_tr = self.zscore_apply(Y_tr, mu, sd)
                Y_te = self.zscore_apply(Y_te, mu, sd)

            w, m = self.fit_lda(Y_tr, y_tr)
            s, g = self.score_samples(Y_te, w, m)
            s_all[val_idx] = s
            g_all[val_idx] = g

        return MetricOutput(
            name=self.name,
            scores={"cids_score": s_all, "cids_margin": g_all},
            directions=self.output_specs,
            cache_state={"cids_features": Y_raw, "cids_score": s_all, "cids_margin": g_all}
        )