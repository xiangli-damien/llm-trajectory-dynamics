import numpy as np
from typing import Dict, List, Optional
from ..core.metrics_base import MetricBase
from ..core.types import MetricDirection, MetricOutput

class RowNull(MetricBase):
    def __init__(self, var_ratio: float = 1.0, tail_window: int = 10):
        super().__init__(var_ratio=var_ratio, tail_window=tail_window)
        self.var_ratio = var_ratio
        self.tail_window = tail_window
        self.eps = 1e-12
    
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
        return {
            'row_len_frac': MetricDirection.HIGHER_BETTER,
            'tail_row_fraction': MetricDirection.HIGHER_BETTER,
            'row_cohere_len': MetricDirection.HIGHER_BETTER,
            'null_cohere_len': MetricDirection.LOWER_BETTER,
            'row_null_gap': MetricDirection.HIGHER_BETTER,
            'null_effective': MetricDirection.LOWER_BETTER,
            'row_frac_gain': MetricDirection.HIGHER_BETTER
        }
    
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
            W = min(self.tail_window, L - 1)
            tail_start = max(0, L - W)
            
            dh = ctx.get_shared_cache('dh')
            if dh is None:
                dh = np.diff(states, axis=1)
            
            tail_dh = dh[:, tail_start:, :]
            
            row_step = np.tensordot(tail_dh, P, axes=([2], [0]))
            null_step = tail_dh - row_step
            
            dh_energy = np.sum(dh ** 2, axis=2)
            row_energy = np.sum(np.tensordot(dh, P, axes=([2], [0])) ** 2, axis=2)
            
            tail_row_energy = np.sum(row_energy[:, tail_start:], axis=1)
            tail_total_energy = np.sum(dh_energy[:, tail_start:], axis=1) + eps
            scores['tail_row_fraction'] = (tail_row_energy / tail_total_energy).astype(np.float32)
            
            len_dh = np.linalg.norm(tail_dh, axis=2)
            len_row = np.linalg.norm(row_step, axis=2)
            len_null = np.linalg.norm(null_step, axis=2)
            
            denom = np.sum(len_dh, axis=1) + eps
            row_len_frac = (np.sum(len_row, axis=1) / denom).astype(np.float32)
            null_len_frac = (np.sum(len_null, axis=1) / denom).astype(np.float32)
            scores['row_len_frac'] = row_len_frac
            
            row_sum = np.sum(row_step, axis=1)
            null_sum = np.sum(null_step, axis=1)
            row_cohere = (np.linalg.norm(row_sum, axis=1) / (np.sum(len_row, axis=1) + eps)).astype(np.float32)
            null_cohere = (np.linalg.norm(null_sum, axis=1) / (np.sum(len_null, axis=1) + eps)).astype(np.float32)
            scores['row_cohere_len'] = row_cohere
            scores['null_cohere_len'] = null_cohere
            
            scores['row_null_gap'] = (row_len_frac - null_len_frac).astype(np.float32)
            scores['null_effective'] = (null_len_frac * null_cohere).astype(np.float32)
            
            ratio = row_energy[:, tail_start:] / (dh_energy[:, tail_start:] + eps)
            K = ratio.shape[1]
            if K >= 2:
                mid = K // 2
                first_half = np.mean(ratio[:, :mid], axis=1)
                last_half = np.mean(ratio[:, mid:], axis=1)
                scores['row_frac_gain'] = (last_half - first_half).astype(np.float32)
            else:
                scores['row_frac_gain'] = np.zeros(N, dtype=np.float32)
        else:
            scores = {
                'row_len_frac': np.zeros(N, dtype=np.float32),
                'tail_row_fraction': np.zeros(N, dtype=np.float32),
                'row_cohere_len': np.zeros(N, dtype=np.float32),
                'null_cohere_len': np.zeros(N, dtype=np.float32),
                'row_null_gap': np.zeros(N, dtype=np.float32),
                'null_effective': np.zeros(N, dtype=np.float32),
                'row_frac_gain': np.zeros(N, dtype=np.float32)
            }
        
        return MetricOutput(
            name=self.name,
            scores=scores,
            directions=self.output_specs,
            cache_state=scores
        )