import numpy as np
from typing import Dict, List
from ..core.metrics_base import MetricBase
from ..core.types import MetricDirection, MetricOutput

class Certainty(MetricBase):
    def __init__(self, top_m: int = 1024, tail_window: int = 3, temperature: float = 1.0):
        super().__init__(top_m=top_m, tail_window=tail_window, temperature=temperature)
        self.top_m = top_m
        self.tail_window = tail_window
        self.temperature = temperature
        self.eps = 1e-12
    
    @property
    def name(self) -> str:
        return "certainty"
    
    @property
    def requires_lm_head(self) -> bool:
        return True
    
    @property
    def supported_modes(self) -> List[str]:
        return ["state"]
    
    @property
    def output_specs(self) -> Dict[str, MetricDirection]:
        return {
            'cert_last_entropy': MetricDirection.HIGHER_BETTER,
            'cert_tail_entropy_slope': MetricDirection.LOWER_BETTER
        }
    
    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        N, L, H = states.shape
        
        last_states = states[:, -1, :]
        last_entropy = np.zeros(N, dtype=np.float32)
        
        for n in range(N):
            logits = ctx.lm_head.compute_logits(last_states[n], temperature=1.0)
            last_entropy[n] = self._compute_entropy_normalized(logits)
        
        tail_entropy_slope = np.zeros(N, dtype=np.float32)
        k = min(self.tail_window, L)
        
        if k >= 2:
            for n in range(N):
                tail_states = states[n, L-k:L, :]
                tail_entropies = np.zeros(k, dtype=np.float32)
                
                for t in range(k):
                    logits = ctx.lm_head.compute_logits(tail_states[t], temperature=1.0)
                    tail_entropies[t] = self._compute_entropy_normalized(logits)
                
                x = np.arange(k, dtype=np.float32)
                y = tail_entropies
                x_mean = x.mean()
                y_mean = y.mean()
                num = np.sum((x - x_mean) * (y - y_mean))
                denom = np.sum((x - x_mean) ** 2)
                slope = num / denom if denom > self.eps else 0.0
                tail_entropy_slope[n] = -slope
        
        scores = {
            'cert_last_entropy': last_entropy,
            'cert_tail_entropy_slope': tail_entropy_slope
        }
        
        return MetricOutput(
            name=self.name,
            scores=scores,
            directions=self.output_specs,
            cache_state=scores
        )
    
    def _compute_entropy_normalized(self, logits: np.ndarray) -> float:
        V = len(logits)
        m = min(self.top_m, V)
        
        if m < V:
            k = V - m
            idx = np.argpartition(logits, k)[k:]
            logits_sel = logits[idx]
        else:
            logits_sel = logits
        
        logits_sel = logits_sel / self.temperature
        logits_max = np.max(logits_sel)
        exp_logits = np.exp(logits_sel - logits_max)
        probs = exp_logits / np.sum(exp_logits)
        
        probs = np.maximum(probs, self.eps)
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(logits_sel))
        
        return 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0