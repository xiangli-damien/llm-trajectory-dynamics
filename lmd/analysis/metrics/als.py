import numpy as np
from typing import Dict, List, Sequence
from ..core.metrics_base import MetricBase
from ..core.types import MetricDirection, MetricOutput
from ..core.utils import auto_tail_window_from_dr

class ALS(MetricBase):
    def __init__(self, tail_window: int = 6, auto_window: bool = True, 
                 top_k: int = 50, temperatures: Sequence[float] = (0.7, 1.0, 1.3), 
                 batch_size_logits: int = 128):
        super().__init__(tail_window=tail_window, auto_window=auto_window, 
                         top_k=top_k, temperatures=tuple(temperatures), 
                         batch_size_logits=batch_size_logits)
        self.tail_window = int(tail_window)
        self.auto_window = bool(auto_window)
        self.top_k = int(top_k)
        self.temperatures = list(temperatures)
        self.batch_size_logits = int(batch_size_logits)
        self.eps = 1e-12

    @property
    def name(self) -> str:
        return "als"

    @property
    def requires_lm_head(self) -> bool:
        return True

    @property
    def supported_modes(self) -> List[str]:
        return ["state"]

    @property
    def output_specs(self) -> Dict[str, MetricDirection]:
        return {"als_stability": MetricDirection.HIGHER_BETTER}

    def _compute_tail_window(self, L: int, ctx) -> int:
        if L <= 1:
            return 1
        if self.auto_window and ('dr' in ctx.shared_cache):
            return min(L, max(2, auto_tail_window_from_dr(ctx.shared_cache['dr'], cover=0.8, min_w=2) + 1))
        return min(self.tail_window, L)

    def _compute_top1_topk(self, lm_head, last_states: np.ndarray) -> tuple:
        N = last_states.shape[0]
        top1 = np.zeros(N, dtype=np.int64)
        topk = np.zeros((N, self.top_k), dtype=np.int64)

        batch_size = max(1, self.batch_size_logits)
        pos = 0
        while pos < N:
            end = min(N, pos + batch_size)
            logits = lm_head.compute_logits(last_states[pos:end])
            
            top1[pos:end] = np.argmax(logits, axis=1)
            
            if self.top_k < logits.shape[1]:
                kth = logits.shape[1] - self.top_k
                part = np.argpartition(logits, kth, axis=1)[:, -self.top_k:]
                part_vals = logits[np.arange(end-pos)[:, None], part]
                order = np.argsort(-part_vals, axis=1)
                topk[pos:end] = part[np.arange(end-pos)[:, None], order]
            else:
                topk[pos:end] = np.tile(np.arange(logits.shape[1], dtype=np.int64), (end-pos, 1))
            pos = end
        return top1, topk

    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        N, L, H = states.shape
        if L < 1:
            zeros = np.zeros(N, dtype=np.float32)
            return MetricOutput(
                name=self.name,
                scores={"als_stability": zeros},
                directions=self.output_specs,
                cache_state={"als_stability": zeros}
            )

        K_layers = self._compute_tail_window(L, ctx)
        layer_start = L - K_layers
        tail_states = states[:, layer_start:, :]

        last_states = states[:, -1, :]
        top1, topk = self._compute_top1_topk(ctx.lm_head, last_states)

        r = ctx.lm_head.rank
        Vh = ctx.lm_head.Vh[:r, :].astype(np.float32, copy=False)
        Y_tail = np.matmul(tail_states, Vh.T).astype(np.float32)

        stability = np.zeros(N, dtype=np.float32)
        n_temperatures = float(len(self.temperatures))

        for n in range(N):
            idx = topk[n]
            if idx.size == 0:
                continue
            agreement_count = 0
            for temperature in self.temperatures:
                z = ctx.lm_head.compute_logits_selected_from_y(Y_tail[n], idx, temperature=temperature)
                pred_idx = idx[np.argmax(z, axis=1)]
                agreement_count += np.sum(pred_idx == top1[n])
            stability[n] = float(agreement_count) / (K_layers * n_temperatures)

        scores = {"als_stability": stability.astype(np.float32)}
        return MetricOutput(
            name=self.name,
            scores=scores,
            directions=self.output_specs,
            cache_state=scores
        )