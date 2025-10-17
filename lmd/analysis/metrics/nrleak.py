import numpy as np
from typing import Dict, List
from ..core.metrics_base import MetricBase
from ..core.types import MetricDirection, MetricOutput
from ..core.utils import auto_tail_window_from_dr

class NRLeak(MetricBase):
    def __init__(self, var_ratio: float = 0.95, tail_window: int = 10, auto_window: bool = True):
        super().__init__(var_ratio=var_ratio, tail_window=tail_window, auto_window=auto_window)
        self.var_ratio = float(var_ratio)
        self.tail_window = int(tail_window)
        self.auto_window = bool(auto_window)
        self.eps = 1e-12

    @property
    def name(self) -> str:
        return "nrleak"

    @property
    def requires_lm_head(self) -> bool:
        return True

    @property
    def supported_modes(self) -> List[str]:
        return ["state"]

    @property
    def output_specs(self) -> Dict[str, MetricDirection]:
        return {
            "nrleak_leak_mean": MetricDirection.LOWER_BETTER,
            "nrleak_qtc": MetricDirection.LOWER_BETTER
        }

    def _compute_tail_window(self, L: int, ctx) -> int:
        if L <= 1:
            return 1
        if self.auto_window and ('dr' in ctx.shared_cache):
            return min(L - 1, max(2, auto_tail_window_from_dr(ctx.shared_cache['dr'], cover=0.8, min_w=2)))
        return min(self.tail_window, L - 1)

    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        N, L, H = states.shape
        if L < 2:
            zeros = np.zeros(N, dtype=np.float32)
            return MetricOutput(
                name=self.name,
                scores={"nrleak_leak_mean": zeros, "nrleak_qtc": zeros},
                directions=self.output_specs,
                cache_state={"nrleak_leak_mean": zeros, "nrleak_qtc": zeros}
            )

        Q = ctx.get_shared_cache('Q')
        if Q is None:
            Q = ctx.lm_head.readout_projection(var_ratio=self.var_ratio)
            ctx.set_shared_cache('Q', Q)
        P = Q @ Q.T
        
        dh = ctx.get_shared_cache('dh')
        if dh is None:
            dh = np.diff(states, axis=1)

        K = self._compute_tail_window(L, ctx)
        start_idx = (L - 1) - (K - 1)
        tail_dh = dh[:, start_idx:, :]

        row_step = np.tensordot(tail_dh, P, axes=([2], [0]))
        null_step = tail_dh - row_step

        if K >= 2:
            n = null_step[:, :-1, :]
            rp1 = row_step[:, 1:, :]
            numerator = np.sum(n * rp1, axis=2)
            denominator = (np.linalg.norm(n, axis=2) * np.linalg.norm(rp1, axis=2)) + self.eps
            leak = numerator / denominator
            leak_mean = np.mean(leak, axis=1).astype(np.float32)
        else:
            leak_mean = np.zeros(N, dtype=np.float32)

        q = ctx.get_shared_cache('q')
        if q is None:
            q = np.tensordot(states, Q, axes=([2], [0]))
            ctx.set_shared_cache('q', q)

        K_layers = min(K + 1, L)
        layer_start = L - K_layers
        q_tail = q[:, layer_start:, :]
        
        if K_layers >= 3:
            dq = np.diff(q_tail, axis=1)
            ddq = np.diff(q_tail, n=2, axis=1)
            arc_length = np.sum(np.linalg.norm(dq, axis=2), axis=1) + self.eps
            curvature = (np.sum(np.linalg.norm(ddq, axis=2), axis=1) / arc_length).astype(np.float32)
        else:
            curvature = np.zeros(N, dtype=np.float32)

        scores = {
            "nrleak_leak_mean": leak_mean,
            "nrleak_qtc": curvature
        }
        return MetricOutput(
            name=self.name,
            scores=scores,
            directions=self.output_specs,
            cache_state=scores
        )