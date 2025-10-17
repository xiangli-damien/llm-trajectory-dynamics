import numpy as np
from typing import Dict, List, Tuple, Optional
from ..core.metrics_base import MetricBase
from ..core.types import MetricDirection, MetricOutput
from .pca_classifier import compute_pca


def compute_time_weights(k: int, mode: str = "exp") -> np.ndarray:
    """Compute time weights for temporal aggregation."""
    idx = np.arange(k, dtype=np.float32)
    if mode == "exp":
        w = np.exp(idx - idx.max())
    elif mode == "linear":
        w = idx + 1.0
    else:
        w = np.ones(k, dtype=np.float32)
    return (w / (w.sum() + 1e-12)).astype(np.float32)


def compute_logpdf_diagonal(Y: np.ndarray, mu: np.ndarray, var: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute log PDF for diagonal covariance Gaussian."""
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


def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize features to zero mean and unit variance."""
    mu = X.mean(0)
    sd = X.std(0) + 1e-12
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_standardization(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Apply standardization using pre-computed statistics."""
    return ((X - mu[None, :]) / sd[None, :]).astype(np.float32)


def _kmeanspp_init(X: np.ndarray, K: int, rng: np.random.RandomState) -> np.ndarray:
    """K-means++ initialization for GMM."""
    N, d = X.shape
    centers = np.empty((K, d), dtype=X.dtype)
    idx = rng.randint(N)
    centers[0] = X[idx]
    closest_dist_sq = np.sum((X - centers[0])**2, axis=1)
    for k in range(1, K):
        probs = closest_dist_sq / (closest_dist_sq.sum() + 1e-12)
        idx = rng.choice(N, p=probs)
        centers[k] = X[idx]
        dist_sq = np.sum((X - centers[k])**2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)
    return centers


def _winsorize(X: np.ndarray, p: float = 0.001) -> np.ndarray:
    """Winsorize data to remove extreme outliers."""
    lo = np.quantile(X, p, axis=0)
    hi = np.quantile(X, 1-p, axis=0)
    return np.clip(X, lo, hi)


def _fit_gmm_diagonal_em(
    X: np.ndarray,
    n_components: int = 2,
    max_iter: int = 100,
    tol: float = 1e-4,
    reg_covar: float = 1e-6,
    seed: int = 42,
    init: str = "kmeans++",
    n_init: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit GMM with diagonal covariance using EM algorithm."""
    rng_master = np.random.RandomState(seed)
    N, d = X.shape
    K = int(n_components)
    
    best = None
    best_ll = -np.inf
    
    for attempt in range(n_init):
        rng = np.random.RandomState(rng_master.randint(0, 1_000_000))
        
        if init == "kmeans++":
            mu = _kmeanspp_init(X, K, rng)
        elif init == "random":
            mu = X[rng.choice(N, K, replace=False)].copy()
        else:
            raise ValueError(f"Unknown init: {init}")
        
        var = np.tile(np.var(X, axis=0) + reg_covar, (K, 1))
        pi = np.ones(K, dtype=np.float32) / K
        
        prev_ll = -np.inf
        for _ in range(max_iter):
            logpdf = compute_logpdf_diagonal(X, mu, var)
            logw = np.log(np.maximum(pi, 1e-12))[None, :] + logpdf
            m = logw.max(axis=1, keepdims=True)
            logsum = m + np.log(np.sum(np.exp(logw - m), axis=1, keepdims=True) + 1e-12)
            gamma = np.exp(logw - logsum)
            
            Nk = gamma.sum(axis=0) + 1e-12
            pi = (Nk / N).astype(np.float32)
            mu = (gamma.T @ X) / Nk[:, None]
            
            for k in range(K):
                diff = X - mu[k][None, :]
                var[k] = (gamma[:, k][:, None] * (diff * diff)).sum(axis=0) / Nk[k]
                var[k] = np.maximum(var[k], reg_covar)
            
            ll = float(logsum.sum())
            if abs(ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
                break
            prev_ll = ll
        
        if ll > best_ll:
            best_ll = ll
            best = (pi.astype(np.float32), mu.astype(np.float32), var.astype(np.float32))
    
    return best


class GMMUnsupervised(MetricBase):
    def __init__(
        self,
        tail_window: int = 10,
        time_weight_mode: str = "exp",
        temporal_mode: str = "aggregate",
        standardize: bool = True,
        shrinkage: float = 1e-2,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed: int = 42,
        feature_source: str = "pca",
        basis_scope: str = "last_k_layers",
        basis_last_k: int = 10,
        pc_rank: int = 32,
        whiten_pca: bool = True,
        use_delta_last: bool = True,
        init: str = "kmeans++",
        n_init: int = 3,
        winsorize_p: float = 0.0,
        fewshot_map_size: int = 20
    ):
        super().__init__(
            tail_window=tail_window,
            time_weight_mode=time_weight_mode,
            temporal_mode=temporal_mode,
            standardize=standardize,
            shrinkage=shrinkage,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
            feature_source=feature_source,
            basis_scope=basis_scope,
            basis_last_k=basis_last_k,
            pc_rank=pc_rank,
            whiten_pca=whiten_pca,
            use_delta_last=use_delta_last,
            init=init,
            n_init=n_init,
            winsorize_p=winsorize_p,
            fewshot_map_size=fewshot_map_size
        )
        self.tail_window = tail_window
        self.time_weight_mode = time_weight_mode
        self.temporal_mode = temporal_mode
        self.standardize = standardize
        self.shrinkage = shrinkage
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.feature_source = feature_source
        self.basis_scope = basis_scope
        self.basis_last_k = basis_last_k
        self.pc_rank = pc_rank
        self.whiten_pca = whiten_pca
        self.use_delta_last = use_delta_last
        self.init = init
        self.n_init = n_init
        self.winsorize_p = winsorize_p
        self.fewshot_map_size = fewshot_map_size
    
    @property
    def name(self) -> str:
        return "gmm_unsup"
    
    @property
    def requires_lm_head(self) -> bool:
        return False
    
    @property
    def supported_modes(self) -> List[str]:
        return ["state"]
    
    @property
    def output_specs(self) -> Dict[str, MetricDirection]:
        return {
            "gmm_unsup_p0": MetricDirection.HIGHER_BETTER,
            "gmm_unsup_p1": MetricDirection.HIGHER_BETTER,
            "gmm_unsup_margin": MetricDirection.HIGHER_BETTER,
            "gmm_unsup_p_correct": MetricDirection.HIGHER_BETTER
        }
    
    def compute_basis(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute PCA basis from states."""
        N, L, H = states.shape
        if self.basis_scope == "global":
            X = states.reshape(N * L, H).astype(np.float32)
        elif self.basis_scope == "last_k_layers":
            k = min(self.basis_last_k, L)
            X = states[:, -k:, :].reshape(N * k, H).astype(np.float32)
        else:
            X = states.reshape(N * L, H).astype(np.float32)
        return compute_pca(X, self.pc_rank)
    
    def project_features(self, states: np.ndarray, basis_representation) -> np.ndarray:
        """Project states using PCA basis."""
        mean, basis, eigenvalues = basis_representation
        N, L, H = states.shape
        k = min(self.tail_window, L)
        
        X_all = states.reshape(N * L, H).astype(np.float32)
        Z_all = ((X_all - mean[None, :]) @ basis).reshape(N, L, -1)
        tail_proj = Z_all[:, -k:, :]
        
        if self.whiten_pca:
            eps = np.sqrt(eigenvalues) + 1e-12
            tail_proj = tail_proj / eps[None, None, :]
        
        if self.temporal_mode in ("poe", "poe_weighted"):
            return tail_proj.astype(np.float32)
        
        w = compute_time_weights(k, self.time_weight_mode)
        mean_rep = np.tensordot(tail_proj, w, axes=([1], [0]))
        
        feats = [mean_rep.astype(np.float32)]
        
        if self.use_delta_last and k >= 2:
            delta_last = (tail_proj[:, -1, :] - tail_proj[:, -2, :]).astype(np.float32)
            feats.append(delta_last)
        
        return np.concatenate(feats, axis=1).astype(np.float32)
    
    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        """Compute GMM unsupervised scores."""
        N, L, H = states.shape
        
        # Extract features
        if self.feature_source == "pca":
            basis = self.compute_basis(states)
            X = self.project_features(states, basis)
        else:
            k = min(self.tail_window, L)
            tail = states[:, -k:, :].astype(np.float32)
            w = compute_time_weights(k, self.time_weight_mode)
            X = np.tensordot(tail, w, axes=([1], [0])).astype(np.float32)
            basis = None
        
        # Winsorize if needed
        if self.winsorize_p > 0:
            X = _winsorize(X, self.winsorize_p)
        
        # Standardize if needed
        if self.standardize:
            mu_z, sd_z = standardize_features(X)
            X_fit = apply_standardization(X, mu_z, sd_z)
        else:
            mu_z, sd_z = None, None
            X_fit = X
        
        # Fit GMM
        pi, mu, var = _fit_gmm_diagonal_em(
            X_fit, n_components=2, max_iter=self.max_iter, tol=self.tol,
            reg_covar=self.shrinkage, seed=self.seed,
            init=self.init, n_init=self.n_init
        )
        
        # Compute posterior probabilities
        logpdf = compute_logpdf_diagonal(X_fit, mu, var)
        logpk = np.log(np.maximum(pi, 1e-12))[None, :]
        logpost = logpk + logpdf
        m = logpost.max(axis=1, keepdims=True)
        logsum = m + np.log(np.sum(np.exp(logpost - m), axis=1, keepdims=True) + 1e-12)
        post = np.exp(logpost - logsum)
        
        p0 = post[:, 0].astype(np.float32)
        p1 = post[:, 1].astype(np.float32)
        margin = np.abs(p1 - 0.5).astype(np.float32)
        
        scores = {
            "gmm_unsup_p0": p0,
            "gmm_unsup_p1": p1,
            "gmm_unsup_margin": margin
        }
        
        # Few-shot mapping for cluster identification
        if self.fewshot_map_size and self.fewshot_map_size > 0:
            rng = np.random.RandomState(self.seed)
            yte = ctx.labels.astype(int)
            idx_all = np.arange(len(yte))
            
            pos = idx_all[yte == 1]
            neg = idx_all[yte == 0]
            if len(pos) > 0 and len(neg) > 0:
                take_pos = min(len(pos), self.fewshot_map_size // 2)
                take_neg = min(len(neg), self.fewshot_map_size - take_pos)
                sel = []
                if take_pos > 0:
                    sel.append(rng.choice(pos, size=take_pos, replace=False))
                if take_neg > 0:
                    sel.append(rng.choice(neg, size=take_neg, replace=False))
                if sel:
                    sel = np.concatenate(sel)
                    ysel = yte[sel]
                    mean_p1_pos = p1[sel][ysel == 1].mean() if np.any(ysel == 1) else None
                    mean_p1_neg = p1[sel][ysel == 0].mean() if np.any(ysel == 0) else None
                    if mean_p1_pos is not None and mean_p1_neg is not None and mean_p1_pos != mean_p1_neg:
                        correct_is_component1 = (mean_p1_pos > mean_p1_neg)
                        scores["gmm_unsup_p_correct"] = (p1 if correct_is_component1 else p0).astype(np.float32)
        
        return MetricOutput(
            name=self.name,
            scores=scores,
            directions=self.output_specs,
            cache_state={
                "pi": pi, "mu": mu, "var": var,
                "standardize_mu": mu_z, "standardize_sd": sd_z,
                "basis": basis
            }
        )