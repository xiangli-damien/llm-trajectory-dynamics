import numpy as np
from typing import Dict, List, Optional, Tuple
from ..core.metrics_base import MetricBase
from ..core.types import MetricDirection, MetricOutput


def compute_time_weights(k: int, mode: str = "exp") -> np.ndarray:
    idx = np.arange(k, dtype=np.float32)
    if mode == "exp":
        w = np.exp(idx - idx.max())
    elif mode == "linear":
        w = idx + 1.0
    else:
        w = np.ones(k, dtype=np.float32)
    return (w / (w.sum() + 1e-12)).astype(np.float32)


def compute_logpdf_diagonal(Y: np.ndarray, mu: np.ndarray, var: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float32)
    mu = np.asarray(mu, dtype=np.float32)
    var = np.asarray(var, dtype=np.float32)
    
    if mu.ndim == 1:
        mu = mu[None, :]
    if var.ndim == 1:
        var = var[None, :]
    d = Y.shape[1]
    var = np.maximum(var, eps)

    logdet = np.sum(np.log(var), axis=1)
    
    diff = Y[:, None, :] - mu[None, :, :]
    quad = np.sum((diff ** 2) / var[None, :, :], axis=2)
    
    logpdf = -0.5 * (d * np.log(2 * np.pi) + logdet[None, :] + quad)
    return logpdf.astype(np.float32)


def fit_single_gaussian(Y: np.ndarray, shrinkage: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    if len(Y) == 0:
        d = Y.shape[1] if Y.ndim > 1 else 1
        return np.zeros(d, dtype=np.float32), np.ones(d, dtype=np.float32)
    mu = Y.mean(0)
    var = Y.var(0) + shrinkage
    return mu.astype(np.float32), var.astype(np.float32)


def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(0)
    sd = X.std(0) + 1e-12
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_standardization(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((X - mu[None, :]) / sd[None, :]).astype(np.float32)


class GMMDiscriminative(MetricBase):
    def __init__(
        self,
        tail_window: int = 10,
        time_weight_mode: str = "exp",
        temporal_mode: str = "aggregate",
        shrinkage: float = 1e-2,
        standardize: bool = True,
        cv_folds: int = 5,
        seed: int = 42
    ):
        super().__init__(
            tail_window=tail_window,
            time_weight_mode=time_weight_mode,
            temporal_mode=temporal_mode,
            shrinkage=shrinkage,
            standardize=standardize,
            cv_folds=cv_folds,
            seed=seed
        )
        self.tail_window = tail_window
        self.time_weight_mode = time_weight_mode
        self.temporal_mode = temporal_mode
        self.shrinkage = shrinkage
        self.standardize = standardize
        self.cv_folds = cv_folds
        self.seed = seed

    @property
    def name(self) -> str:
        return "gmm_disc"

    @property
    def requires_lm_head(self) -> bool:
        return False

    @property
    def supported_modes(self) -> List[str]:
        return ["state"]

    @property
    def output_specs(self) -> Dict[str, MetricDirection]:
        return {"gmm_llr": MetricDirection.HIGHER_BETTER,
                "gmm_posterior": MetricDirection.HIGHER_BETTER}

    def extract_tail_features(self, ctx, states: np.ndarray) -> np.ndarray:
        N, L, H = states.shape
        k = min(self.tail_window, L)
        return states[:, -k:, :].astype(np.float32)

    def aggregate_temporal_features(self, tail_features: np.ndarray) -> np.ndarray:
        N, k, d = tail_features.shape
        w = compute_time_weights(k, self.time_weight_mode)
        mean_rep = np.tensordot(tail_features, w, axes=([1], [0]))
        return mean_rep.astype(np.float32)

    def fit_class_gaussians(self, Y_train: np.ndarray, y_train: np.ndarray):
        Y_pos = Y_train[y_train == 1]
        Y_neg = Y_train[y_train == 0]
        mu_pos, var_pos = fit_single_gaussian(Y_pos, self.shrinkage)
        mu_neg, var_neg = fit_single_gaussian(Y_neg, self.shrinkage)
        return (mu_pos, var_pos), (mu_neg, var_neg)

    def compute_llr(self, Y_test: np.ndarray,
                    pos_params: Tuple[np.ndarray, np.ndarray],
                    neg_params: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        mu_pos, var_pos = pos_params
        mu_neg, var_neg = neg_params
        logp_pos = compute_logpdf_diagonal(Y_test, mu_pos, var_pos)
        logp_neg = compute_logpdf_diagonal(Y_test, mu_neg, var_neg)
        if logp_pos.ndim > 1:
            logp_pos = logp_pos.max(1)
            logp_neg = logp_neg.max(1)
        llr = logp_pos - logp_neg
        posterior = 1.0 / (1.0 + np.exp(-llr))
        return llr.astype(np.float32), posterior.astype(np.float32)

    def fit_score_poe(self, Q_train: np.ndarray, y_train: np.ndarray,
                      Q_test: np.ndarray, weighted: bool) -> Tuple[np.ndarray, np.ndarray, dict]:
        N_test, k, d = Q_test.shape
        llr_sum = np.zeros(N_test, dtype=np.float32)
        weights = compute_time_weights(k, self.time_weight_mode) if weighted else np.ones(k, dtype=np.float32)
        per_time_params = []
        
        for t in range(k):
            Y_tr_t = Q_train[:, t, :]
            Y_te_t = Q_test[:, t, :]
            if self.standardize:
                mu, sd = standardize_features(Y_tr_t)
                Y_tr_t = apply_standardization(Y_tr_t, mu, sd)
                Y_te_t = apply_standardization(Y_te_t, mu, sd)
            pos_params, neg_params = self.fit_class_gaussians(Y_tr_t, y_train)
            llr_t, _ = self.compute_llr(Y_te_t, pos_params, neg_params)
            llr_sum += (weights[t] * llr_t)
            per_time_params.append({
                "t": int(t),
                "mu_pos": pos_params[0], "var_pos": pos_params[1],
                "mu_neg": neg_params[0], "var_neg": neg_params[1]
            })
        
        posterior = (1.0 / (1.0 + np.exp(-llr_sum))).astype(np.float32)
        return llr_sum, posterior, {"per_time_params": per_time_params, "weights": weights}

    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        labels = ctx.labels.astype(int)
        N = len(labels)
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return MetricOutput(name=self.name,
                                scores={"gmm_llr": np.zeros(N, dtype=np.float32),
                                        "gmm_posterior": np.full(N, 0.5, dtype=np.float32)},
                                directions=self.output_specs)
        
        Q = self.extract_tail_features(ctx, states)
        rng = np.random.RandomState(self.seed)
        n_folds = min(self.cv_folds, len(pos_idx), len(neg_idx))
        llr_all = np.zeros(N, dtype=np.float32)
        posterior_all = np.full(N, 0.5, dtype=np.float32)
        
        if n_folds < 2:
            if self.temporal_mode == "aggregate":
                Y = self.aggregate_temporal_features(Q)
                if self.standardize:
                    mu, sd = standardize_features(Y)
                    Y = apply_standardization(Y, mu, sd)
                pos_params, neg_params = self.fit_class_gaussians(Y, labels)
                llr, posterior = self.compute_llr(Y, pos_params, neg_params)
                llr_all, posterior_all = llr, posterior
                cache = {"mu_pos": pos_params[0], "var_pos": pos_params[1],
                         "mu_neg": neg_params[0], "var_neg": neg_params[1]}
            else:
                weighted = (self.temporal_mode == "poe_weighted")
                llr, posterior, cache = self.fit_score_poe(Q, labels, Q, weighted)
                llr_all, posterior_all = llr, posterior
        else:
            rng.shuffle(pos_idx)
            rng.shuffle(neg_idx)
            pos_splits = np.array_split(pos_idx, n_folds)
            neg_splits = np.array_split(neg_idx, n_folds)
            cache = None
            
            for kf in range(n_folds):
                val_idx = np.concatenate([pos_splits[kf], neg_splits[kf]])
                tr_idx = np.concatenate([
                    np.concatenate([pos_splits[i] for i in range(n_folds) if i != kf]),
                    np.concatenate([neg_splits[i] for i in range(n_folds) if i != kf])
                ])
                Qtr, Qte = Q[tr_idx], Q[val_idx]
                ytr = labels[tr_idx]
                
                if self.temporal_mode == "aggregate":
                    Ytr = self.aggregate_temporal_features(Qtr)
                    Yte = self.aggregate_temporal_features(Qte)
                    if self.standardize:
                        mu, sd = standardize_features(Ytr)
                        Ytr = apply_standardization(Ytr, mu, sd)
                        Yte = apply_standardization(Yte, mu, sd)
                    pos_params, neg_params = self.fit_class_gaussians(Ytr, ytr)
                    llr, posterior = self.compute_llr(Yte, pos_params, neg_params)
                else:
                    weighted = (self.temporal_mode == "poe_weighted")
                    llr, posterior, _ = self.fit_score_poe(Qtr, ytr, Qte, weighted)
                
                llr_all[val_idx] = llr
                posterior_all[val_idx] = posterior
        
        return MetricOutput(
            name=self.name,
            scores={"gmm_llr": llr_all, "gmm_posterior": posterior_all},
            directions=self.output_specs,
            cache_state=cache
        )