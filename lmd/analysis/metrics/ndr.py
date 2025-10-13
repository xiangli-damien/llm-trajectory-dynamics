import numpy as np
from typing import Dict, List, Optional
from ..core.metrics_base import MetricBase
from ..core.types import MetricDirection, MetricOutput

class NDR(MetricBase):
    def __init__(self, tail_window: int = 10, compute_curvature: bool = False,
                 drop_weak_keys: bool = True):
        super().__init__(tail_window=tail_window, compute_curvature=compute_curvature,
                         drop_weak_keys=drop_weak_keys)
        self.tail_window = tail_window
        self.compute_curvature = compute_curvature
        self.drop_weak_keys = drop_weak_keys
        self.eps = 1e-12
        self.weak_keys = {"ndr_forward_rate", "ndr_back_rate"}
    
    @property
    def name(self) -> str:
        return "ndr"
    
    @property
    def requires_lm_head(self) -> bool:
        return False
    
    @property
    def supported_modes(self) -> List[str]:
        return ["state"]
    
    @property
    def output_specs(self) -> Dict[str, MetricDirection]:
        specs = {
            'ndr': MetricDirection.HIGHER_BETTER,
            'ndr_lognorm_slope': MetricDirection.HIGHER_BETTER,
            'ndr_last2_gap': MetricDirection.HIGHER_BETTER,
            'ndr_last2_ratio': MetricDirection.HIGHER_BETTER,
            'ndr_zigzag': MetricDirection.HIGHER_BETTER,
            'ndr_backtrack_net': MetricDirection.HIGHER_BETTER,
            'ndr_orth_to_net': MetricDirection.HIGHER_BETTER,
            'ndr_proj_variance': MetricDirection.LOWER_BETTER
        }
        
        if not self.drop_weak_keys:
            specs['ndr_forward_rate'] = MetricDirection.HIGHER_BETTER
            specs['ndr_back_rate'] = MetricDirection.LOWER_BETTER
        
        if self.compute_curvature:
            specs['ndr_tail_curvature'] = MetricDirection.LOWER_BETTER
        
        return specs
    
    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        N, L, H = states.shape
        eps = self.eps
        
        norms = ctx.get_shared_cache('norms')
        lognorms = ctx.get_shared_cache('lognorms')
        
        if norms is None:
            norms = np.linalg.norm(states, axis=2)
            lognorms = np.log(np.maximum(norms, eps))
        
        mean_norms = norms.mean(axis=1)
        final_norms = norms[:, -1]
        ndr = mean_norms / (final_norms + eps)
        
        W = min(self.tail_window, L)
        tail_start = max(0, L - W)
        tail_lognorms = lognorms[:, tail_start:]
        
        slopes = self._compute_slopes(tail_lognorms)
        
        scores = {
            'ndr': ndr.astype(np.float32),
            'ndr_lognorm_slope': slopes
        }
        
        if L >= 2:
            norm_diffs = np.abs(np.diff(norms, axis=1))
            last2_diff = norm_diffs[:, -1]
            total_diff = np.sum(norm_diffs, axis=1)
            scores['ndr_last2_gap'] = last2_diff.astype(np.float32)
            scores['ndr_last2_ratio'] = (last2_diff / (total_diff + eps)).astype(np.float32)
            
            dh = ctx.get_shared_cache('dh')
            if dh is None:
                dh = np.diff(states, axis=1)
            
            diff_tail_start = max(0, (L - 1) - (W - 1))
            tail_dh = dh[:, diff_tail_start:, :]
            
            d_total = states[:, -1, :] - states[:, tail_start, :]
            d_norm = np.linalg.norm(d_total, axis=1) + eps
            d_hat = d_total / d_norm[:, None]
            
            path_num = np.linalg.norm(np.sum(tail_dh, axis=1), axis=1)
            path_den = np.sum(np.linalg.norm(tail_dh, axis=2), axis=1) + eps
            path_eff = path_num / path_den
            scores['ndr_zigzag'] = (1.0 - path_eff).astype(np.float32)
            
            proj = np.sum(tail_dh * d_hat[:, None, :], axis=2)
            proj_abs_sum = np.sum(np.abs(proj), axis=1) + eps
            scores['ndr_backtrack_net'] = (np.sum(np.maximum(0.0, -proj), axis=1) / proj_abs_sum).astype(np.float32)
            
            perp = tail_dh - proj[:, :, None] * d_hat[:, None, :]
            perp_sum = np.sum(np.linalg.norm(perp, axis=2), axis=1)
            dh_sum = np.sum(np.linalg.norm(tail_dh, axis=2), axis=1) + eps
            scores['ndr_orth_to_net'] = (perp_sum / dh_sum).astype(np.float32)
            
            scores['ndr_proj_variance'] = np.var(proj, axis=1).astype(np.float32)
            
            if not self.drop_weak_keys:
                forward_mask = (proj > 0).astype(np.float32)
                back_mask = (proj < 0).astype(np.float32)
                scores['ndr_forward_rate'] = np.mean(forward_mask, axis=1).astype(np.float32)
                scores['ndr_back_rate'] = np.mean(back_mask, axis=1).astype(np.float32)
        else:
            scores['ndr_zigzag'] = np.zeros(N, dtype=np.float32)
            scores['ndr_backtrack_net'] = np.zeros(N, dtype=np.float32)
            scores['ndr_orth_to_net'] = np.zeros(N, dtype=np.float32)
            scores['ndr_proj_variance'] = np.zeros(N, dtype=np.float32)
            if not self.drop_weak_keys:
                scores['ndr_forward_rate'] = np.ones(N, dtype=np.float32)
                scores['ndr_back_rate'] = np.zeros(N, dtype=np.float32)
        
        if self.compute_curvature and tail_lognorms.shape[1] >= 3:
            dd = np.diff(tail_lognorms, n=2, axis=1)
            scores['ndr_tail_curvature'] = np.sqrt(np.mean(dd ** 2, axis=1)).astype(np.float32)
        
        return MetricOutput(
            name=self.name,
            scores=scores,
            directions=self.output_specs,
            cache_state=scores
        )
    
    def _compute_slopes(self, log_norms: np.ndarray) -> np.ndarray:
        N, k = log_norms.shape
        slopes = np.zeros(N, dtype=np.float32)
        if k >= 2:
            x = np.arange(k, dtype=np.float32)
            x0 = x - x.mean()
            denom = np.sum(x0 * x0) + self.eps
            y0 = log_norms - log_norms.mean(axis=1, keepdims=True)
            slopes = (np.sum(y0 * x0, axis=1) / denom).astype(np.float32)
        return -slopes