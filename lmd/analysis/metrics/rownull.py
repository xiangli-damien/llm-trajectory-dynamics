import numpy as np
from typing import Dict, List, Optional
from ..core.metrics_base import MetricBase
from ..core.types import MetricDirection, MetricOutput

class RowNull(MetricBase):
    def __init__(self, var_ratio: float = 0.95, tail_window: int = 10,
                 drop_weak_keys: bool = True):
        super().__init__(var_ratio=var_ratio, tail_window=tail_window,
                         drop_weak_keys=drop_weak_keys)
        self.var_ratio = var_ratio
        self.tail_window = tail_window
        self.drop_weak_keys = drop_weak_keys
        self.eps = 1e-12
        self.weak_keys = {"row_forward_rate", "row_back_rate", "null_forward_rate",
                         "row_energy_slope", "null_energy_slope"}
    
    @property
    def name(self) -> str:
        return "rownull"
    
    @property
    def requires_lm_head(self) -> bool:
        return True
    
    @property
    def supported_modes(self) -> List[str]:
        return ["state"]
    
    @property
    def output_specs(self) -> Dict[str, MetricDirection]:
        specs = {
            'row_fraction': MetricDirection.HIGHER_BETTER,
            'tail_row_fraction': MetricDirection.HIGHER_BETTER,
            'row_len_frac': MetricDirection.HIGHER_BETTER,
            'null_len_frac': MetricDirection.LOWER_BETTER,
            'row_cohere_len': MetricDirection.LOWER_BETTER,
            'null_cohere_len': MetricDirection.LOWER_BETTER,
            'tail_null_fraction': MetricDirection.LOWER_BETTER,
            'rownull_ratio_slope': MetricDirection.HIGHER_BETTER
        }
        
        if not self.drop_weak_keys:
            specs.update({
                'row_forward_rate': MetricDirection.HIGHER_BETTER,
                'row_back_rate': MetricDirection.LOWER_BETTER,
                'null_forward_rate': MetricDirection.LOWER_BETTER,
                'row_energy_slope': MetricDirection.HIGHER_BETTER,
                'null_energy_slope': MetricDirection.LOWER_BETTER
            })
        
        return specs
    
    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        N, L, H = states.shape
        eps = self.eps
        
        Q = ctx.get_shared_cache('Q')
        if Q is None:
            Q = ctx.lm_head.readout_projection(var_ratio=self.var_ratio)
            ctx.set_shared_cache('Q', Q)
        
        P = Q @ Q.T
        
        scores = {}
        
        if L > 1:
            W = min(self.tail_window, L)
            tail_start = max(0, L - W)
            
            dh = np.diff(states, axis=1)
            tail_dh = dh[:, tail_start:, :]
            
            row_step = np.tensordot(tail_dh, P, axes=([2], [0]))
            null_step = tail_dh - row_step
            
            dh_energy = np.sum(dh ** 2, axis=2)
            row_energy = np.sum(np.tensordot(dh, P, axes=([2], [0])) ** 2, axis=2)
            total_energy = dh_energy + eps
            scores['row_fraction'] = (np.sum(row_energy, axis=1) / np.sum(total_energy, axis=1)).astype(np.float32)
            
            tail_row_energy = np.sum(row_energy[:, tail_start:], axis=1)
            tail_total_energy = np.sum(total_energy[:, tail_start:], axis=1)
            scores['tail_row_fraction'] = (tail_row_energy / tail_total_energy).astype(np.float32)
            
            len_dh = np.linalg.norm(tail_dh, axis=2)
            len_row = np.linalg.norm(row_step, axis=2)
            len_null = np.linalg.norm(null_step, axis=2)
            
            denom = np.sum(len_dh, axis=1) + eps
            scores['row_len_frac'] = (np.sum(len_row, axis=1) / denom).astype(np.float32)
            scores['null_len_frac'] = (np.sum(len_null, axis=1) / denom).astype(np.float32)
            
            row_sum = np.sum(row_step, axis=1)
            null_sum = np.sum(null_step, axis=1)
            scores['row_cohere_len'] = (np.linalg.norm(row_sum, axis=1) / (np.sum(len_row, axis=1) + eps)).astype(np.float32)
            scores['null_cohere_len'] = (np.linalg.norm(null_sum, axis=1) / (np.sum(len_null, axis=1) + eps)).astype(np.float32)
            
            tail_null_energy = np.sum((tail_dh - row_step) ** 2, axis=(1, 2))
            tail_total_energy_sum = np.sum(tail_dh ** 2, axis=(1, 2)) + eps
            scores['tail_null_fraction'] = np.clip(tail_null_energy / tail_total_energy_sum, 0.0, 1.0).astype(np.float32)
            
            ratio = row_energy[:, tail_start:] / (dh_energy[:, tail_start:] + eps)
            K = ratio.shape[1]
            if K >= 2:
                x = (np.arange(K, dtype=np.float32) - (K-1)/2.0)
                denom_slope = np.sum(x*x) + eps
                ratio_centered = ratio - ratio.mean(axis=1, keepdims=True)
                scores['rownull_ratio_slope'] = (np.sum(ratio_centered * x[None, :], axis=1) / denom_slope).astype(np.float32)
            else:
                scores['rownull_ratio_slope'] = np.zeros(N, dtype=np.float32)
            
            if not self.drop_weak_keys:
                d_total = states[:, -1, :] - states[:, tail_start, :]
                d_total_row = d_total @ P
                d_row_norm = np.linalg.norm(d_total_row, axis=1) + eps
                d_hat_row = d_total_row / d_row_norm[:, None]
                d_total_null = d_total - d_total_row
                d_null_norm = np.linalg.norm(d_total_null, axis=1) + eps
                d_hat_null = d_total_null / d_null_norm[:, None]
                
                row_proj = np.sum(row_step * d_hat_row[:, None, :], axis=2)
                null_proj = np.sum(null_step * d_hat_null[:, None, :], axis=2)
                scores['row_forward_rate'] = np.mean((row_proj > 0).astype(np.float32), axis=1).astype(np.float32)
                scores['row_back_rate'] = np.mean((row_proj < 0).astype(np.float32), axis=1).astype(np.float32)
                scores['null_forward_rate'] = np.mean((null_proj > 0).astype(np.float32), axis=1).astype(np.float32)
                
                len_dh_safe = len_dh + eps
                row_ratio = len_row / len_dh_safe
                null_ratio = len_null / len_dh_safe
                if K >= 2:
                    x2 = np.arange(K, dtype=np.float32)
                    xm = x2 - x2.mean()
                    denom2 = np.sum(xm * xm) + eps
                    scores['row_energy_slope'] = (np.sum((row_ratio - row_ratio.mean(axis=1, keepdims=True)) * xm[None, :], axis=1) / denom2).astype(np.float32)
                    scores['null_energy_slope'] = (np.sum((null_ratio - null_ratio.mean(axis=1, keepdims=True)) * xm[None, :], axis=1) / denom2).astype(np.float32)
                else:
                    scores['row_energy_slope'] = np.zeros(N, dtype=np.float32)
                    scores['null_energy_slope'] = np.zeros(N, dtype=np.float32)
        else:
            base_scores = {
                'row_fraction': np.zeros(N, dtype=np.float32),
                'tail_row_fraction': np.zeros(N, dtype=np.float32),
                'row_len_frac': np.zeros(N, dtype=np.float32),
                'null_len_frac': np.zeros(N, dtype=np.float32),
                'row_cohere_len': np.zeros(N, dtype=np.float32),
                'null_cohere_len': np.zeros(N, dtype=np.float32),
                'tail_null_fraction': np.ones(N, dtype=np.float32),
                'rownull_ratio_slope': np.zeros(N, dtype=np.float32)
            }
            scores = base_scores
            
            if not self.drop_weak_keys:
                scores.update({
                    'row_forward_rate': np.ones(N, dtype=np.float32),
                    'row_back_rate': np.zeros(N, dtype=np.float32),
                    'null_forward_rate': np.zeros(N, dtype=np.float32),
                    'row_energy_slope': np.zeros(N, dtype=np.float32),
                    'null_energy_slope': np.zeros(N, dtype=np.float32)
                })
        
        return MetricOutput(
            name=self.name,
            scores=scores,
            directions=self.output_specs,
            cache_state=scores
        )