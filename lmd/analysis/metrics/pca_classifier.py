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


def compute_pca(X: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_centered = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    rank = min(rank, Vt.shape[0])
    mean = X.mean(0).astype(np.float32)
    basis = Vt[:rank, :].T.astype(np.float32)
    eigenvalues = (S[:rank] ** 2).astype(np.float32)
    return mean, basis, eigenvalues


def regularized_lda(
    Y: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.2,
    gamma: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray]:
    Y_pos = Y[labels == 1]
    Y_neg = Y[labels == 0]
    d = Y.shape[1]
    
    if len(Y_pos) == 0 or len(Y_neg) == 0:
        return np.zeros(d, dtype=np.float32), np.zeros(d, dtype=np.float32)
    
    mu_pos = Y_pos.mean(0)
    mu_neg = Y_neg.mean(0)
    
    n_pos, n_neg = len(Y_pos), len(Y_neg)
    C_pos = np.cov(Y_pos.T, bias=False) if n_pos > 1 else np.eye(d, dtype=np.float32)
    C_neg = np.cov(Y_neg.T, bias=False) if n_neg > 1 else np.eye(d, dtype=np.float32)
    
    S_within = ((n_pos - 1) * C_pos + (n_neg - 1) * C_neg) / max(1, n_pos + n_neg - 2)
    identity = np.eye(d, dtype=np.float32)
    S_regularized = (1.0 - alpha) * S_within + alpha * identity + gamma * identity
    
    try:
        weight = np.linalg.solve(S_regularized, mu_pos - mu_neg)
    except Exception:
        weight = mu_pos - mu_neg
    
    center = 0.5 * (mu_pos + mu_neg)
    return weight.astype(np.float32), center.astype(np.float32)


class PCABasisClassifier(MetricBase):
    def __init__(
        self,
        tail_window: int = 10,
        basis_scope: str = "global",
        basis_last_k: int = 10,
        pc_rank: int = 64,
        time_weight_mode: str = "exp",
        use_delta_last: bool = True,
        whiten_pca: bool = True,
        rda_alpha: float = 0.2,
        gamma_regularization: float = 1e-3,
        standardize: bool = True,
        cv_folds: int = 5,
        seed: int = 42
    ):
        super().__init__(
            tail_window=tail_window,
            basis_scope=basis_scope,
            basis_last_k=basis_last_k,
            pc_rank=pc_rank,
            time_weight_mode=time_weight_mode,
            use_delta_last=use_delta_last,
            whiten_pca=whiten_pca,
            rda_alpha=rda_alpha,
            gamma_regularization=gamma_regularization,
            standardize=standardize,
            cv_folds=cv_folds,
            seed=seed
        )
        self.tail_window = tail_window
        self.basis_scope = basis_scope
        self.basis_last_k = basis_last_k
        self.pc_rank = pc_rank
        self.time_weight_mode = time_weight_mode
        self.use_delta_last = use_delta_last
        self.whiten_pca = whiten_pca
        self.rda_alpha = rda_alpha
        self.gamma_regularization = gamma_regularization
        self.standardize = standardize
        self.cv_folds = cv_folds
        self.seed = seed

    @property
    def name(self) -> str:
        return "pca_cls"

    @property
    def requires_lm_head(self) -> bool:
        return False

    @property
    def supported_modes(self) -> List[str]:
        return ["state"]

    @property
    def output_specs(self) -> Dict[str, MetricDirection]:
        return {
            "pca_cls_score":  MetricDirection.HIGHER_BETTER,
            "pca_cls_margin": MetricDirection.HIGHER_BETTER
        }

    def compute_basis(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N, L, H = states.shape
        
        if self.basis_scope == "global":
            X = states.reshape(N * L, H).astype(np.float32)
        elif self.basis_scope == "last_k_layers":
            k = min(self.basis_last_k, L)
            tail_states = states[:, -k:, :]
            X = tail_states.reshape(N * k, H).astype(np.float32)
        else:
            X = states.reshape(N * L, H).astype(np.float32)
        
        return compute_pca(X, self.pc_rank)

    def project_features(
        self,
        states: np.ndarray,
        basis_representation
    ) -> np.ndarray:
        N, L, H = states.shape
        k = min(self.tail_window, L)
        mean, basis, eigenvalues = basis_representation
        
        X_all = states.reshape(N * L, H).astype(np.float32)
        Z_all = ((X_all - mean[None, :]) @ basis).reshape(N, L, -1)
        tail_proj = Z_all[:, -k:, :]
        
        if self.whiten_pca:
            eps = np.sqrt(eigenvalues) + 1e-12
            tail_proj = tail_proj / eps[None, None, :]
        
        w = compute_time_weights(k, self.time_weight_mode)
        mean_rep = np.tensordot(tail_proj, w, axes=([1], [0]))
        
        feats = [mean_rep.astype(np.float32)]
        
        if self.use_delta_last and k >= 2:
            delta_last = (tail_proj[:, -1, :] - tail_proj[:, -2, :]).astype(np.float32)
            feats.append(delta_last)
        
        return np.concatenate(feats, axis=1).astype(np.float32)

    def standardize_data(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = X.mean(0)
        sd = X.std(0) + 1e-12
        return mu.astype(np.float32), sd.astype(np.float32)

    def apply_standardization(self, X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
        return ((X - mu[None, :]) / sd[None, :]).astype(np.float32)

    def compute_scores(
        self,
        Y: np.ndarray,
        weight: np.ndarray,
        center: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = (Y - center[None, :]) @ weight
        margins = np.abs(scores) / (np.linalg.norm(weight) + 1e-12)
        return scores.astype(np.float32), margins.astype(np.float32)

    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        labels = ctx.labels.astype(int)
        N = len(labels)
        
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return MetricOutput(
                name=self.name,
                scores={"pca_cls_score": np.zeros(N, dtype=np.float32),
                        "pca_cls_margin": np.zeros(N, dtype=np.float32)},
                directions=self.output_specs
            )
        
        rng = np.random.RandomState(self.seed)
        n_folds = min(self.cv_folds, len(pos_idx), len(neg_idx))
        
        scores_all = np.zeros(N, dtype=np.float32)
        margins_all = np.zeros(N, dtype=np.float32)
        
        if n_folds < 2:
            # No CV, use all data for training and testing
            basis = self.compute_basis(states)
            Y_raw = self.project_features(states, basis)
            
            Y = Y_raw
            if self.standardize:
                mu, sd = self.standardize_data(Y)
                Y = self.apply_standardization(Y, mu, sd)
            w, c = regularized_lda(Y, labels, self.rda_alpha, self.gamma_regularization)
            s, g = self.compute_scores(Y, w, c)
            scores_all, margins_all = s, g
        else:
            # Cross-validation with proper train/val split for basis computation
            rng.shuffle(pos_idx)
            rng.shuffle(neg_idx)
            pos_splits = np.array_split(pos_idx, n_folds)
            neg_splits = np.array_split(neg_idx, n_folds)
            
            for kf in range(n_folds):
                val_idx = np.concatenate([pos_splits[kf], neg_splits[kf]])
                tr_idx = np.concatenate([
                    np.concatenate([pos_splits[i] for i in range(n_folds) if i != kf]),
                    np.concatenate([neg_splits[i] for i in range(n_folds) if i != kf])
                ])
                
                # Compute basis ONLY on training fold
                states_tr = states[tr_idx]
                basis = self.compute_basis(states_tr)
                
                # Project both train and validation data using train basis
                Y_tr = self.project_features(states[tr_idx], basis)
                Y_te = self.project_features(states[val_idx], basis)
                
                y_tr = labels[tr_idx]
                
                if self.standardize:
                    mu, sd = self.standardize_data(Y_tr)
                    Y_tr = self.apply_standardization(Y_tr, mu, sd)
                    Y_te = self.apply_standardization(Y_te, mu, sd)
                
                w, c = regularized_lda(Y_tr, y_tr, self.rda_alpha, self.gamma_regularization)
                s, g = self.compute_scores(Y_te, w, c)
                scores_all[val_idx] = s
                margins_all[val_idx] = g
        
        return MetricOutput(
            name=self.name,
            scores={"pca_cls_score": scores_all, "pca_cls_margin": margins_all},
            directions=self.output_specs
        )