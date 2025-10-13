import numpy as np
from typing import Dict, List
from ..core.metrics_base import MetricBase
from ..core.types import MetricDirection, MetricOutput

class CoE(MetricBase):
    def __init__(self):
        super().__init__()
    
    @property
    def name(self) -> str:
        return "coe"
    
    @property
    def requires_lm_head(self) -> bool:
        return False
    
    @property
    def supported_modes(self) -> List[str]:
        return ["state"]
    
    @property
    def output_specs(self) -> Dict[str, MetricDirection]:
        return {
            'coe_r': MetricDirection.HIGHER_BETTER,
            'coe_c': MetricDirection.HIGHER_BETTER
        }
    
    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        N, L, H = states.shape
        if L < 2:
            scores = {
                'coe_r': np.zeros(N, dtype=np.float32),
                'coe_c': np.zeros(N, dtype=np.float32)
            }
        else:
            coe_r, coe_c = self._compute_coe_scores(states)
            scores = {
                'coe_r': coe_r,
                'coe_c': coe_c
            }
        
        return MetricOutput(
            name=self.name,
            scores=scores,
            directions=self.output_specs,
            cache_state=scores
        )
    
    def _compute_coe_scores(self, states: np.ndarray):
        N, L, H = states.shape
        
        h_first = states[:, 0, :]
        h_last = states[:, -1, :]
        
        mag_diff = h_last - h_first
        mag_denom_raw = np.linalg.norm(mag_diff, axis=1, keepdims=True)
        mag_denom = np.where(mag_denom_raw > 1e-6, mag_denom_raw, 1.0)
        
        cos_angle = np.sum(h_last * h_first, axis=1) / (
            np.linalg.norm(h_last, axis=1) * np.linalg.norm(h_first, axis=1) + 1e-8
        )
        ang_denom_raw = np.arccos(np.clip(cos_angle, -1, 1))
        ang_denom = np.where(ang_denom_raw > 1e-6, ang_denom_raw, 1.0)
        
        diffs = states[:, 1:, :] - states[:, :-1, :]
        mags = np.linalg.norm(diffs, axis=2) / mag_denom
        
        h_curr = states[:, 1:, :]
        h_prev = states[:, :-1, :]
        dots = np.sum(h_curr * h_prev, axis=2)
        norms_curr = np.linalg.norm(h_curr, axis=2) + 1e-8
        norms_prev = np.linalg.norm(h_prev, axis=2) + 1e-8
        cos_angles = dots / (norms_curr * norms_prev)
        angles_raw = np.arccos(np.clip(cos_angles, -1, 1))
        
        mask = (ang_denom_raw > 1e-6)[:, np.newaxis]
        angles = np.where(mask, angles_raw / ang_denom[:, None], angles_raw)
        
        coe_r = np.mean(mags, axis=1) - np.mean(angles, axis=1)
        x_coords = mags * np.cos(angles)
        y_coords = mags * np.sin(angles)
        coe_c = np.sqrt(np.mean(x_coords, axis=1)**2 + np.mean(y_coords, axis=1)**2)
        
        return coe_r.astype(np.float32), coe_c.astype(np.float32)