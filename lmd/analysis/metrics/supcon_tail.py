import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def extract_tail_representation(ctx, states, tail_window: int):
    N, L, H = states.shape
    k = min(tail_window, L)
    return states[:, -k:, :].astype(np.float32)


def build_feature_vector(
    q_tail: np.ndarray,
    feature_mode: str,
    time_weight_mode: str,
    use_delta_last: bool = True
) -> np.ndarray:
    N, k, r = q_tail.shape
    
    if feature_mode == "flat":
        return q_tail.reshape(N, k * r)
    
    if feature_mode == "mean_delta":
        w = compute_time_weights(k, time_weight_mode)
        mean_rep = np.tensordot(q_tail, w, axes=([1], [0]))
        
        if use_delta_last and k >= 2:
            delta_last = q_tail[:, -1, :] - q_tail[:, -2, :]
            return np.concatenate([mean_rep, delta_last], axis=1).astype(np.float32)
        else:
            return mean_rep.astype(np.float32)
    
    return q_tail.reshape(N, k * r)


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
        return d1 - d0


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


class SupConTail(MetricBase):
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
        device: str = "cpu"
    ):
        super().__init__(
            tail_window=tail_window,
            feature_mode=feature_mode,
            time_weight_mode=time_weight_mode,
            use_delta_last=use_delta_last,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            temperature=temperature,
            lambda_ce=lambda_ce,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            augment_drop_prob=augment_drop_prob,
            augment_noise_std=augment_noise_std,
            seed=seed,
            device=device
        )
        
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
    
    @property
    def name(self) -> str:
        return "supcon_tail"
    
    @property
    def requires_lm_head(self) -> bool:
        return False
    
    @property
    def supported_modes(self) -> List[str]:
        return ["state"]
    
    @property
    def output_specs(self) -> Dict[str, MetricDirection]:
        return {"supcon_logit": MetricDirection.HIGHER_BETTER}
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray
    ) -> np.ndarray:
        torch.manual_seed(self.seed)
        dev = torch.device(self.device)
        
        X_train_t = torch.from_numpy(X_train).to(dev)
        y_train_t = torch.from_numpy(y_train.astype(np.int64)).to(dev)
        X_test_t = torch.from_numpy(X_test).to(dev)
        
        in_dim = X_train.shape[1]
        
        encoder = ResidualMLP(in_dim, self.hidden_dim, self.embed_dim, self.dropout).to(dev)
        head = PrototypeHead(self.embed_dim).to(dev)
        
        params = list(encoder.parameters()) + list(head.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        
        supcon_loss = SupConLoss(self.temperature)
        
        bsz = min(self.batch_size, X_train.shape[0])
        N_train = X_train.shape[0]
        
        for epoch in range(self.epochs):
            perm = torch.randperm(N_train, device=dev)
            
            current_temp = max(0.05, self.temperature * (0.95 ** epoch))
            supcon_loss.temperature = current_temp
            
            for start in range(0, N_train, bsz):
                idx = perm[start:start + bsz]
                x_batch = X_train_t[idx]
                y_batch = y_train_t[idx]
                
                v1, v2 = apply_augmentation(
                    x_batch,
                    self.augment_drop_prob,
                    self.augment_noise_std
                )
                
                z1 = encoder(v1)
                z2 = encoder(v2)
                
                z_combined = torch.cat([z1, z2], dim=0)
                loss_contrastive = supcon_loss(z_combined, y_batch)
                
                logit = head(z1)
                loss_classification = F.binary_cross_entropy_with_logits(
                    logit,
                    y_batch.float()
                )
                
                loss = (1.0 - self.lambda_ce) * loss_contrastive + self.lambda_ce * loss_classification
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        with torch.no_grad():
            z_test = encoder(X_test_t)
            scores = head(z_test)
        
        return scores.detach().cpu().numpy().astype(np.float32)
    
    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        labels = ctx.labels.astype(int)
        N = len(labels)
        
        q_tail = extract_tail_representation(ctx, states, self.tail_window)
        X = build_feature_vector(q_tail, self.feature_mode, self.time_weight_mode, self.use_delta_last)
        
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return MetricOutput(
                name=self.name,
                scores={"supcon_logit": np.zeros(N, dtype=np.float32)},
                directions=self.output_specs
            )
        
        n_folds = min(5, len(pos_idx), len(neg_idx))
        rng = np.random.RandomState(self.seed)
        
        scores_all = np.zeros(N, dtype=np.float32)
        
        if n_folds < 2:
            scores_all = self.train_model(X, labels, X)
        else:
            rng.shuffle(pos_idx)
            rng.shuffle(neg_idx)
            
            pos_splits = np.array_split(pos_idx, n_folds)
            neg_splits = np.array_split(neg_idx, n_folds)
            
            for fold_idx in range(n_folds):
                val_idx = np.concatenate([pos_splits[fold_idx], neg_splits[fold_idx]])
                train_idx = np.concatenate([
                    np.concatenate([pos_splits[i] for i in range(n_folds) if i != fold_idx]),
                    np.concatenate([neg_splits[i] for i in range(n_folds) if i != fold_idx])
                ])
                
                scores = self.train_model(X[train_idx], labels[train_idx], X[val_idx])
                scores_all[val_idx] = scores
        
        return MetricOutput(
            name=self.name,
            scores={"supcon_logit": scores_all},
            directions=self.output_specs
        )