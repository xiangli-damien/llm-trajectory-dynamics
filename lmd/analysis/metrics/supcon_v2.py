import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os
from ..core.metrics_base import MetricBase
from ..core.types import MetricDirection, MetricOutput
from ..core.utils import auto_tail_window_from_dr
from .pca_classifier import compute_pca, regularized_lda


def extract_tail_representation(ctx, states: np.ndarray, tail_window: int) -> np.ndarray:
    N, L, H = states.shape
    k = min(tail_window, L)
    return states[:, -k:, :].astype(np.float32)


def compute_time_weights(k: int, mode: str = "exp") -> np.ndarray:
    idx = np.arange(k, dtype=np.float32)
    if mode == "exp":
        w = np.exp(idx - idx.max())
    elif mode == "linear":
        w = idx + 1.0
    else:
        w = np.ones(k, dtype=np.float32)
    return (w / (w.sum() + 1e-12)).astype(np.float32)


def build_feature_vector(
    tail_features: np.ndarray,
    feature_mode: str,
    time_weight_mode: str,
    use_delta_last: bool = True
) -> np.ndarray:
    N, k, d = tail_features.shape
    
    if feature_mode == "flat":
        return tail_features.reshape(N, k * d)
    
    if feature_mode == "mean_delta":
        w = compute_time_weights(k, time_weight_mode)
        mean_rep = np.tensordot(tail_features, w, axes=([1], [0]))
        
        if use_delta_last and k >= 2:
            delta_last = tail_features[:, -1, :] - tail_features[:, -2, :]
            return np.concatenate([mean_rep, delta_last], axis=1).astype(np.float32)
        else:
            return mean_rep.astype(np.float32)
    
    return tail_features.reshape(N, k * d)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.ln1(h)
        h = self.dropout(h)
        
        residual = h
        h = F.relu(self.fc2(h))
        h = self.ln2(h + residual)
        h = self.dropout(h)
        
        z = self.fc3(h)
        return F.normalize(z, p=2, dim=-1)


class PrototypeHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.c0 = nn.Parameter(torch.randn(embed_dim))
        self.c1 = nn.Parameter(torch.randn(embed_dim))
        
    def forward(self, z):
        c0 = F.normalize(self.c0, p=2, dim=0)
        c1 = F.normalize(self.c1, p=2, dim=0)
        d0 = ((z - c0) ** 2).sum(dim=1)
        d1 = ((z - c1) ** 2).sum(dim=1)
        return d0 - d1


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z: torch.Tensor, y: torch.Tensor):
        z = F.normalize(z, p=2, dim=-1)
        B2 = z.size(0)
        
        if y.dim() == 1:
            y = y.view(-1, 1)
        if y.size(0) * 2 == B2:
            y = torch.cat([y, y], dim=0)
        
        sim = torch.mm(z, z.t()) / self.temperature
        logits = sim - sim.max(dim=1, keepdim=True).values.detach()
        
        mask = torch.eq(y, y.t()).float()
        mask.fill_diagonal_(0.0)
        
        eye = torch.eye(B2, device=z.device)
        exp_logits = torch.exp(logits) * (1.0 - eye)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        pos_count = mask.sum(1, keepdim=True).clamp_min(1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1, keepdim=True) / pos_count
        
        return -mean_log_prob_pos.mean()


def apply_augmentation(x: torch.Tensor, drop_prob: float, noise_std: float):
    B, D = x.shape
    device = x.device
    
    mask1 = (torch.rand(B, D, device=device) > drop_prob).float()
    mask2 = (torch.rand(B, D, device=device) > drop_prob).float()
    
    v1 = x * mask1 + noise_std * torch.randn_like(x)
    v2 = x * mask2 + noise_std * torch.randn_like(x)
    
    return v1, v2


class SupConV2(MetricBase):
    def __init__(
        self,
        tail_window: int = 10,
        feature_mode: str = "mean_delta",
        time_weight_mode: str = "exp",
        use_delta_last: bool = True,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        temperature: float = 0.1,
        lambda_ce: float = 0.5,
        epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        augment_drop_prob: float = 0.0,
        augment_noise_std: float = 0.0,
        seed: int = 42,
        device: str = "cpu",
        use_readout_q: bool = False,
        var_ratio: float = 0.95,
        pre_pca_rank: int = 0,
        pre_pca_whiten: bool = True,
        standardize: bool = False,
        balanced_batch: bool = True,
        proto_momentum: float = 0.0,
        lda_align_weight: float = 0.0,
        aux_scalar_keys: list = None,
        load_checkpoint: str = None,
        save_checkpoint: str = None,
        freeze_encoder: bool = False
    ):
        super().__init__()
        self.tail_window = tail_window
        self.feature_mode = feature_mode
        self.time_weight_mode = time_weight_mode
        self.use_delta_last = use_delta_last
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.lambda_ce = lambda_ce
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.augment_drop_prob = augment_drop_prob
        self.augment_noise_std = augment_noise_std
        self.seed = seed
        self.device = device
        self.use_readout_q = use_readout_q
        self.var_ratio = float(var_ratio)
        self.pre_pca_rank = int(pre_pca_rank)
        self.pre_pca_whiten = bool(pre_pca_whiten)
        self.standardize = bool(standardize)
        self.balanced_batch = bool(balanced_batch)
        self.proto_momentum = float(proto_momentum)
        self.lda_align_weight = float(lda_align_weight)
        self.aux_scalar_keys = aux_scalar_keys or []
        self.load_checkpoint = load_checkpoint
        self.save_checkpoint = save_checkpoint
        self.freeze_encoder = bool(freeze_encoder)
    
    @property
    def name(self) -> str:
        return "supcon_v2"
    
    @property
    def requires_lm_head(self) -> bool:
        return bool(self.use_readout_q)
    
    @property
    def supported_modes(self) -> List[str]:
        return ["state"]
    
    @property
    def output_specs(self) -> Dict[str, MetricDirection]:
        return {"supcon_v2_logit": MetricDirection.HIGHER_BETTER}
    
    def _extract_features(self, ctx, states: np.ndarray) -> np.ndarray:
        if self.use_readout_q and (ctx.lm_head is not None):
            Q = ctx.lm_head.readout_projection(var_ratio=self.var_ratio)
            k = min(self.tail_window, states.shape[1])
            tail_q = np.tensordot(states[:, -k:, :], Q, axes=([2],[0])).astype(np.float32)
        else:
            tail_q = extract_tail_representation(ctx, states, self.tail_window)
        
        X = build_feature_vector(tail_q, self.feature_mode, self.time_weight_mode, self.use_delta_last)
        
        if self.aux_scalar_keys:
            scalars = []
            candidate_states = ['coe', 'ndr', 'dac', 'als', 'certainty', 'cids', 'gmm_disc']
            for key in self.aux_scalar_keys:
                for name in candidate_states:
                    st = ctx.get_metric_state(name) or {}
                    if key in st:
                        v = np.asarray(st[key]).reshape(-1, 1).astype(np.float32)
                        scalars.append(v)
                        break
            if scalars:
                S = np.concatenate(scalars, axis=1).astype(np.float32)
                mu = S.mean(0)
                sd = S.std(0) + 1e-12
                S = (S - mu[None, :]) / sd[None, :]
                X = np.concatenate([X, S], axis=1)
        
        return X
    
    def _preprocess_features(self, Xtr: np.ndarray, Xte: np.ndarray):
        if self.pre_pca_rank > 0:
            mean, basis, evals = compute_pca(Xtr, self.pre_pca_rank)
            Xtr = (Xtr - mean[None, :]) @ basis
            Xte = (Xte - mean[None, :]) @ basis
            if self.pre_pca_whiten:
                eps = np.sqrt(evals[:self.pre_pca_rank]) + 1e-12
                Xtr = Xtr / eps[None, :]
                Xte = Xte / eps[None, :]
        if self.standardize:
            mu = Xtr.mean(0)
            sd = Xtr.std(0) + 1e-12
            Xtr = (Xtr - mu[None, :]) / sd[None, :]
            Xte = (Xte - mu[None, :]) / sd[None, :]
        return Xtr.astype(np.float32), Xte.astype(np.float32)
    
    def train_model_direct(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        torch.manual_seed(self.seed)
        dev = torch.device(self.device)
        
        Xtr, Xte = self._preprocess_features(X_train, X_test)
        
        Xtr_t = torch.from_numpy(Xtr).to(dev)
        ytr_t = torch.from_numpy(y_train.astype(np.int64)).to(dev)
        Xte_t = torch.from_numpy(Xte).to(dev)
        
        in_dim = Xtr.shape[1]
        encoder = ResidualMLP(in_dim, self.hidden_dim, self.embed_dim, self.dropout).to(dev)
        head = PrototypeHead(self.embed_dim).to(dev)
        
        if self.load_checkpoint and os.path.exists(self.load_checkpoint):
            state = torch.load(self.load_checkpoint, map_location=dev)
            if 'encoder' in state:
                encoder.load_state_dict(state['encoder'])
            if 'head' in state:
                head.load_state_dict(state['head'])
            if self.freeze_encoder:
                for p in encoder.parameters():
                    p.requires_grad = False
        
        params = list(filter(lambda p: p.requires_grad, encoder.parameters())) + list(head.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        supcon_loss = SupConLoss(self.temperature)
        
        with torch.no_grad():
            z_all = encoder(Xtr_t)
            if (y_train == 1).any():
                c1_init = F.normalize(z_all[ytr_t == 1].mean(0), dim=0)
                head.c1.copy_(c1_init)
            if (y_train == 0).any():
                c0_init = F.normalize(z_all[ytr_t == 0].mean(0), dim=0)
                head.c0.copy_(c0_init)
        
        if self.epochs == 0:
            with torch.no_grad():
                z_test = encoder(Xte_t)
                scores = head(z_test)
            if self.save_checkpoint:
                torch.save({'encoder': encoder.state_dict(), 'head': head.state_dict()}, self.save_checkpoint)
            return scores.detach().cpu().numpy().astype(np.float32)
        
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        rng = np.random.RandomState(self.seed)
        
        bsz = min(self.batch_size, Xtr.shape[0])
        
        for epoch in range(self.epochs):
            supcon_loss.temperature = max(0.05, self.temperature * (0.95 ** epoch))
            
            w_t, mu1_all, mu0_all = None, None, None
            if self.lda_align_weight > 0:
                with torch.no_grad():
                    z_all = encoder(Xtr_t)
                    if (y_train == 1).any() and (y_train == 0).any():
                        mu1_all = z_all[ytr_t == 1].mean(0)
                        mu0_all = z_all[ytr_t == 0].mean(0)
                        w, _ = regularized_lda(z_all.detach().cpu().numpy(), y_train, alpha=0.2, gamma=1e-3)
                        w_t = torch.from_numpy(w).to(dev)
            
            n_steps = int(np.ceil(Xtr.shape[0] / bsz))
            for _ in range(n_steps):
                if self.balanced_batch and len(pos_idx) > 0 and len(neg_idx) > 0:
                    p = min(len(pos_idx), bsz // 2)
                    n = bsz - p
                    sel = np.concatenate([
                        rng.choice(pos_idx, size=p, replace=True),
                        rng.choice(neg_idx, size=n, replace=True)
                    ])
                else:
                    sel = rng.choice(np.arange(Xtr.shape[0]), size=bsz, replace=False)
                
                x_batch = Xtr_t[sel]
                y_batch = ytr_t[sel]
                
                v1, v2 = apply_augmentation(x_batch, self.augment_drop_prob, self.augment_noise_std)
                z1 = encoder(v1)
                z2 = encoder(v2)
                
                z_combined = torch.cat([z1, z2], dim=0)
                loss_contrastive = supcon_loss(z_combined, y_batch)
                
                logit = head(z1)
                loss_ce = F.binary_cross_entropy_with_logits(logit, y_batch.float())
                
                loss = (1.0 - self.lambda_ce) * loss_contrastive + self.lambda_ce * loss_ce
                
                if (w_t is not None) and (mu1_all is not None) and (mu0_all is not None):
                    align = 1.0 - F.cosine_similarity(mu1_all - mu0_all, w_t, dim=0)
                    loss = loss + self.lda_align_weight * align
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                z_all = encoder(Xtr_t)
                if (y_train == 1).any():
                    c1_new = F.normalize(z_all[ytr_t == 1].mean(0), dim=0)
                    if self.proto_momentum > 0:
                        head.c1.copy_(self.proto_momentum * head.c1 + (1 - self.proto_momentum) * c1_new)
                    else:
                        head.c1.copy_(c1_new)
                if (y_train == 0).any():
                    c0_new = F.normalize(z_all[ytr_t == 0].mean(0), dim=0)
                    if self.proto_momentum > 0:
                        head.c0.copy_(self.proto_momentum * head.c0 + (1 - self.proto_momentum) * c0_new)
                    else:
                        head.c0.copy_(c0_new)
        
        with torch.no_grad():
            z_test = encoder(Xte_t)
            scores = head(z_test)
        
        if self.save_checkpoint:
            torch.save({'encoder': encoder.state_dict(), 'head': head.state_dict()}, self.save_checkpoint)
        
        return scores.detach().cpu().numpy().astype(np.float32)
    
    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        labels = ctx.labels.astype(int)
        N = len(labels)
        
        X_raw = self._extract_features(ctx, states)
        
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return MetricOutput(
                name=self.name,
                scores={"supcon_v2_logit": np.zeros(N, dtype=np.float32)},
                directions=self.output_specs
            )
        
        n_folds = min(5, len(pos_idx), len(neg_idx))
        rng = np.random.RandomState(self.seed)
        scores_all = np.zeros(N, dtype=np.float32)
        
        if n_folds < 2:
            scores_all = self.train_model_direct(X_raw, labels, X_raw)
        else:
            rng.shuffle(pos_idx)
            rng.shuffle(neg_idx)
            pos_splits = np.array_split(pos_idx, n_folds)
            neg_splits = np.array_split(neg_idx, n_folds)
            for k in range(n_folds):
                val_idx = np.concatenate([pos_splits[k], neg_splits[k]])
                tr_idx = np.concatenate([
                    np.concatenate([pos_splits[i] for i in range(n_folds) if i != k]),
                    np.concatenate([neg_splits[i] for i in range(n_folds) if i != k])
                ])
                scores = self.train_model_direct(X_raw[tr_idx], labels[tr_idx], X_raw[val_idx])
                scores_all[val_idx] = scores
        
        return MetricOutput(
            name=self.name,
            scores={"supcon_v2_logit": scores_all},
            directions=self.output_specs
        )