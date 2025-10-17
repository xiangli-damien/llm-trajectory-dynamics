import numpy as np
from typing import Dict, List
from ..core.metrics_base import MetricBase
from ..core.types import MetricDirection, MetricOutput
from ..core.utils import auto_tail_window_from_dr

class DAC(MetricBase):
    def __init__(self, tail_window: int = 10, auto_window: bool = True, 
                 winsor_q: float = 0.01, batch_size_logits: int = 128):
        super().__init__(tail_window=tail_window, auto_window=auto_window, 
                         winsor_q=winsor_q, batch_size_logits=batch_size_logits)
        self.tail_window = int(tail_window)
        self.auto_window = bool(auto_window)
        self.winsor_q = float(winsor_q)
        self.batch_size_logits = int(batch_size_logits)
        self.eps = 1e-12

    @property
    def name(self) -> str:
        return "dac"

    @property
    def requires_lm_head(self) -> bool:
        return True

    @property
    def supported_modes(self) -> List[str]:
        return ["state"]

    @property
    def output_specs(self) -> Dict[str, MetricDirection]:
        return {
            "dac_posfrac": MetricDirection.HIGHER_BETTER,
            "dac_cohere": MetricDirection.HIGHER_BETTER,
            "dac_fliprate": MetricDirection.LOWER_BETTER
        }

    def _compute_tail_window(self, L: int, ctx) -> int:
        if L <= 1:
            return 1
        if self.auto_window and ('dr' in ctx.shared_cache):
            return min(L - 1, max(2, auto_tail_window_from_dr(ctx.shared_cache['dr'], cover=0.8, min_w=2)))
        return min(self.tail_window, L - 1)

    def _compute_top2(self, lm_head, last_states: np.ndarray) -> tuple:
        N = last_states.shape[0]
        top1 = np.zeros(N, dtype=np.int64)
        top2 = np.zeros(N, dtype=np.int64)
        batch_size = max(1, self.batch_size_logits)
        
        pos = 0
        while pos < N:
            end = min(N, pos + batch_size)
            logits = lm_head.compute_logits(last_states[pos:end])
            part = np.argpartition(logits, -2, axis=1)[:, -2:]
            vals = logits[np.arange(end - pos)[:, None], part]
            order = np.argsort(-vals, axis=1)
            top_sorted = part[np.arange(end - pos)[:, None], order]
            top1[pos:end] = top_sorted[:, 0]
            top2[pos:end] = top_sorted[:, 1]
            pos = end
        return top1, top2

    def _winsorize_data(self, x: np.ndarray, q: float) -> np.ndarray:
        if x.ndim == 2 and x.shape[1] >= 3:
            lo = np.quantile(x, q, axis=1, keepdims=True)
            hi = np.quantile(x, 1.0 - q, axis=1, keepdims=True)
            return np.clip(x, lo, hi)
        return x

    def _compute_fliprate(self, x: np.ndarray) -> np.ndarray:
        N, K = x.shape
        if K < 2:
            return np.zeros(N, dtype=np.float32)
        
        signs = np.sign(x + 1e-7)
        for i in range(N):
            for t in range(1, K):
                if signs[i, t] == 0:
                    signs[i, t] = signs[i, t-1]
        
        flips = np.sum(signs[:, 1:] * signs[:, :-1] < 0, axis=1).astype(np.float32)
        return (flips / (K - 1.0)).astype(np.float32)

    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        N, L, H = states.shape
        if L < 2:
            zeros = np.zeros(N, dtype=np.float32)
            return MetricOutput(
                name=self.name,
                scores={"dac_posfrac": zeros, "dac_cohere": zeros, "dac_fliprate": zeros},
                directions=self.output_specs,
                cache_state={"dac_posfrac": zeros, "dac_cohere": zeros, "dac_fliprate": zeros}
            )

        K = self._compute_tail_window(L, ctx)
        start_idx = (L - 1) - (K - 1)

        r = ctx.lm_head.rank
        U = ctx.lm_head.U[:, :r].astype(np.float32, copy=False)
        S = ctx.lm_head.S[:r].astype(np.float32, copy=False)
        Vh = ctx.lm_head.Vh[:r, :].astype(np.float32, copy=False)
        US = (U * S[None, :]).astype(np.float32)

        last_states = states[:, -1, :]
        y_top1, y_top2 = self._compute_top2(ctx.lm_head, last_states)
        US_delta = (US[y_top1, :] - US[y_top2, :]).astype(np.float32)

        dh = ctx.get_shared_cache('dh')
        if dh is None:
            dh = np.diff(states, axis=1)
        tail_dh = dh[:, start_idx:, :]
        Y = np.matmul(tail_dh, Vh.T)

        DeltaM = np.sum(Y * US_delta[:, None, :], axis=2).astype(np.float32)

        if self.winsor_q > 0:
            DeltaM = self._winsorize_data(DeltaM, self.winsor_q)

        posfrac = np.mean((DeltaM > 0).astype(np.float32), axis=1).astype(np.float32)
        denom = np.sum(np.abs(DeltaM), axis=1) + self.eps
        cohere = (np.abs(np.sum(DeltaM, axis=1)) / denom).astype(np.float32)
        fliprate = self._compute_fliprate(DeltaM)

        scores = {
            "dac_posfrac": posfrac,
            "dac_cohere": cohere,
            "dac_fliprate": fliprate
        }
        return MetricOutput(
            name=self.name,
            scores=scores,
            directions=self.output_specs,
            cache_state=scores
        )