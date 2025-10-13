import numpy as np
from typing import Dict, List, Optional, Tuple
from ..core.metrics_base import MetricBase
from ..core.types import MetricDirection, MetricOutput

def tail_weighted_q(ctx, states, tail_window: int, var_ratio: float):
    N, L, H = states.shape
    k = min(tail_window, L)
    Q = ctx.get_shared_cache('Q')
    if Q is None:
        Q = ctx.lm_head.readout_projection(var_ratio=var_ratio)
        ctx.set_shared_cache('Q', Q)
    q = np.tensordot(states, Q, axes=([2],[0]))
    q_tail = q[:, -k:, :]
    idx = np.arange(k, dtype=np.float32)
    w = np.exp(idx - idx.max())
    w = (w / (w.sum() + 1e-12)).astype(np.float32)
    x = np.tensordot(q_tail, w, axes=([1],[0]))
    return x.astype(np.float32)

def zscore_fit(X):
    mu = X.mean(0)
    sd = X.std(0) + 1e-12
    return mu.astype(np.float32), sd.astype(np.float32)

def zscore_apply(X, mu, sd):
    return ((X - mu[None, :]) / sd[None, :]).astype(np.float32)

def shrink_cov(S, gamma=1e-6):
    d = S.shape[0]
    return (S + gamma * np.eye(d, dtype=np.float32)).astype(np.float32)

class TailFDA(MetricBase):
    def __init__(self, tail_window: int = 10, var_ratio: float = 0.95,
                 k_subspace: int = 2, gamma: float = 1e-3, cv_folds: int = 5,
                 seed: int = 42, standardize: bool = True):
        super().__init__(tail_window=tail_window, var_ratio=var_ratio,
                         k_subspace=k_subspace, gamma=gamma, cv_folds=cv_folds,
                         seed=seed, standardize=standardize)
        self.tail_window = tail_window
        self.var_ratio = var_ratio
        self.k_subspace = k_subspace
        self.gamma = gamma
        self.cv_folds = cv_folds
        self.seed = seed
        self.standardize = standardize

    @property
    def name(self):
        return "subspace_fda"
    
    @property
    def requires_lm_head(self):
        return True
    
    @property
    def supported_modes(self):
        return ["state"]
    
    @property
    def output_specs(self):
        return {"twfda_score": MetricDirection.HIGHER_BETTER}

    def learn_B(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        x1, x0 = X[y==1], X[y==0]
        mu1, mu0 = x1.mean(0), x0.mean(0)
        mu = X.mean(0)
        C1 = np.cov(x1.T, bias=False) if len(x1) > 1 else np.eye(X.shape[1], dtype=np.float32)
        C0 = np.cov(x0.T, bias=False) if len(x0) > 1 else np.eye(X.shape[1], dtype=np.float32)
        Sw = ((len(x1)-1)*C1 + (len(x0)-1)*C0) / max(1, (n-2))
        Sw = shrink_cov(Sw, self.gamma)
        Sb = (len(x1) * np.outer(mu1-mu, mu1-mu) + len(x0) * np.outer(mu0-mu, mu0-mu)).astype(np.float32)
        try:
            A = np.linalg.solve(Sw, Sb)
        except:
            A = np.linalg.pinv(Sw) @ Sb
        w, V = np.linalg.eigh(A)
        idx = np.argsort(w)[::-1]
        V = V[:, idx]
        k = min(self.k_subspace, V.shape[1])
        return V[:, :k].astype(np.float32)

    def lda_1d(self, Z: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu1, mu0 = Z[y==1].mean(0), Z[y==0].mean(0)
        n1, n0 = (y==1).sum(), (y==0).sum()
        C1 = np.cov(Z[y==1].T, bias=False) if n1 > 1 else np.eye(Z.shape[1], dtype=np.float32)
        C0 = np.cov(Z[y==0].T, bias=False) if n0 > 1 else np.eye(Z.shape[1], dtype=np.float32)
        Sw = ((n1-1)*C1 + (n0-1)*C0) / max(1, (n1 + n0 - 2))
        Sw = shrink_cov(Sw, self.gamma)
        try:
            w = np.linalg.solve(Sw, (mu1 - mu0))
        except:
            w = (mu1 - mu0)
        m = 0.5 * (mu1 + mu0)
        return w.astype(np.float32), m.astype(np.float32)

    def score(self, Z: np.ndarray, w: np.ndarray, m: np.ndarray):
        s = (Z - m[None, :]) @ w
        return s.astype(np.float32)

    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        X_raw = tail_weighted_q(ctx, states, self.tail_window, self.var_ratio)
        y = ctx.labels.astype(int)
        N = len(y)
        rng = np.random.RandomState(self.seed)
        pos = np.where(y==1)[0]
        neg = np.where(y==0)[0]
        if len(pos)==0 or len(neg)==0:
            return MetricOutput(name=self.name,
                scores={"twfda_score": np.zeros(N, dtype=np.float32)},
                directions=self.output_specs,
                cache_state={"X_tail": X_raw, "twfda_score": np.zeros(N, dtype=np.float32)})
        folds = min(self.cv_folds, len(pos), len(neg))
        if folds < 2:
            X = X_raw
            if self.standardize:
                mu, sd = zscore_fit(X)
                X = zscore_apply(X, mu, sd)
            B = self.learn_B(X, y)
            Z = X @ B
            w, m = self.lda_1d(Z, y)
            s = self.score(Z, w, m)
            return MetricOutput(name=self.name,
                scores={"twfda_score": s},
                directions=self.output_specs,
                cache_state={"X_tail": X_raw, "B": B, "twfda_score": s})
        rng.shuffle(pos)
        rng.shuffle(neg)
        pos_s = np.array_split(pos, folds)
        neg_s = np.array_split(neg, folds)
        s_all = np.zeros(N, dtype=np.float32)
        for k in range(folds):
            val = np.concatenate([pos_s[k], neg_s[k]])
            tr = np.concatenate([np.concatenate([pos_s[i] for i in range(folds) if i!=k]),
                                 np.concatenate([neg_s[i] for i in range(folds) if i!=k])])
            Xtr, Xte = X_raw[tr], X_raw[val]
            ytr = y[tr]
            if self.standardize:
                mu, sd = zscore_fit(Xtr)
                Xtr = zscore_apply(Xtr, mu, sd)
                Xte = zscore_apply(Xte, mu, sd)
            B = self.learn_B(Xtr, ytr)
            Ztr, Zte = Xtr @ B, Xte @ B
            w, m = self.lda_1d(Ztr, ytr)
            s_all[val] = self.score(Zte, w, m)
        return MetricOutput(name=self.name,
            scores={"twfda_score": s_all},
            directions=self.output_specs,
            cache_state={"X_tail": X_raw, "twfda_score": s_all})

class SPCASupervised(MetricBase):
    def __init__(self, tail_window: int = 10, var_ratio: float = 0.95,
                 k_pc: int = 4, gamma: float = 1e-3, cv_folds: int = 5,
                 seed: int = 42, standardize: bool = True):
        super().__init__(tail_window=tail_window, var_ratio=var_ratio,
                         k_pc=k_pc, gamma=gamma, cv_folds=cv_folds, seed=seed,
                         standardize=standardize)
        self.tail_window = tail_window
        self.var_ratio = var_ratio
        self.k_pc = k_pc
        self.gamma = gamma
        self.cv_folds = cv_folds
        self.seed = seed
        self.standardize = standardize

    @property
    def name(self):
        return "spca_sup"
    
    @property
    def requires_lm_head(self):
        return True
    
    @property
    def supported_modes(self):
        return ["state"]
    
    @property
    def output_specs(self):
        return {"spca_sup_score": MetricDirection.HIGHER_BETTER}

    def whiten_by_within(self, X: np.ndarray, y: np.ndarray):
        x1, x0 = X[y==1], X[y==0]
        n1, n0 = len(x1), len(x0)
        C1 = np.cov(x1.T, bias=False) if n1>1 else np.eye(X.shape[1], dtype=np.float32)
        C0 = np.cov(x0.T, bias=False) if n0>1 else np.eye(X.shape[1], dtype=np.float32)
        Sw = ((n1-1)*C1 + (n0-1)*C0) / max(1, (n1+n0-2))
        U, S, Vt = np.linalg.svd(Sw + self.gamma*np.eye(Sw.shape[0], dtype=np.float32), full_matrices=False)
        W = (U @ np.diag(1.0/np.sqrt(S)) @ U.T).astype(np.float32)
        return W

    def select_pcs(self, Xw: np.ndarray, y: np.ndarray):
        Xc = Xw - Xw.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        V = Vt.T.astype(np.float32)
        mu1 = Xw[y==1].mean(0)
        mu0 = Xw[y==0].mean(0)
        dmu = (mu1 - mu0).astype(np.float32)
        alpha = np.abs(V.T @ dmu)
        idx = np.argsort(-alpha)[:self.k_pc]
        Vsel = V[:, idx]
        signs = np.sign((Vsel.T @ dmu)).astype(np.float32)
        weights = alpha[idx].astype(np.float32)
        return Vsel, signs, weights

    def score_pool(self, Xw: np.ndarray, Vsel: np.ndarray, signs: np.ndarray, weights: np.ndarray):
        Z = Xw @ Vsel
        Zs = Z * signs[None, :]
        s = (Zs * (weights[None, :] / (weights.sum() + 1e-12))).sum(axis=1)
        return s.astype(np.float32)

    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        X_raw = tail_weighted_q(ctx, states, self.tail_window, self.var_ratio)
        y = ctx.labels.astype(int)
        N = len(y)
        rng = np.random.RandomState(self.seed)
        pos = np.where(y==1)[0]
        neg = np.where(y==0)[0]
        if len(pos)==0 or len(neg)==0:
            return MetricOutput(name=self.name,
                scores={"spca_sup_score": np.zeros(N, dtype=np.float32)},
                directions=self.output_specs,
                cache_state={"X_tail": X_raw, "spca_sup_score": np.zeros(N, dtype=np.float32)})
        folds = min(self.cv_folds, len(pos), len(neg))
        if folds < 2:
            X = X_raw
            W = self.whiten_by_within(X, y)
            Xw = (X - X.mean(0, keepdims=True)) @ W
            Vsel, signs, weights = self.select_pcs(Xw, y)
            s = self.score_pool(Xw, Vsel, signs, weights)
            return MetricOutput(name=self.name,
                scores={"spca_sup_score": s},
                directions=self.output_specs,
                cache_state={"X_tail": X_raw, "Vsel": Vsel, "spca_sup_score": s})
        rng.shuffle(pos)
        rng.shuffle(neg)
        pos_s = np.array_split(pos, folds)
        neg_s = np.array_split(neg, folds)
        s_all = np.zeros(N, dtype=np.float32)
        for k in range(folds):
            val = np.concatenate([pos_s[k], neg_s[k]])
            tr = np.concatenate([np.concatenate([pos_s[i] for i in range(folds) if i!=k]),
                                 np.concatenate([neg_s[i] for i in range(folds) if i!=k])])
            Xtr, Xte = X_raw[tr], X_raw[val]
            ytr = y[tr]
            W = self.whiten_by_within(Xtr, ytr)
            mu = Xtr.mean(0, keepdims=True)
            Xtr_w = (Xtr - mu) @ W
            Xte_w = (Xte - mu) @ W
            Vsel, signs, weights = self.select_pcs(Xtr_w, ytr)
            s_all[val] = self.score_pool(Xte_w, Vsel, signs, weights)
        return MetricOutput(name=self.name,
            scores={"spca_sup_score": s_all},
            directions=self.output_specs,
            cache_state={"X_tail": X_raw, "spca_sup_score": s_all})