import numpy as np
from typing import Dict, List
from ..core.metrics_base import MetricBase
from ..core.types import MetricDirection, MetricOutput

def rank_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    noise = 1e-12 * np.random.RandomState(0).randn(*x.shape).astype(np.float32)
    x = x + noise
    xs = np.sort(x)
    idx = np.searchsorted(xs, x, side='right')
    return (idx.astype(np.float32) / (len(x) + 1e-12)).astype(np.float32)

class FusionRankAvg(MetricBase):
    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "fusion_rankavg"
    
    @property
    def requires_lm_head(self) -> bool:
        return False
    
    @property
    def supported_modes(self) -> List[str]:
        return ["state"]
    
    @property
    def output_specs(self) -> Dict[str, MetricDirection]:
        return {"fusion_rankavg": MetricDirection.HIGHER_BETTER}
    
    @property
    def dependencies(self) -> List[str]:
        return ["cids", "gmm_disc"]
    
    def compute_state(self, ctx, states) -> MetricOutput:
        N = states.shape[0]
        cids_state = ctx.get_metric_state('cids') or {}
        gmm_state = ctx.get_metric_state('gmm_disc') or {}
        s1 = cids_state.get("cids_score", None)
        s2 = gmm_state.get("gmm_llr", None)
        if s1 is None or s2 is None:
            return MetricOutput(
                name=self.name,
                scores={"fusion_rankavg": np.zeros(N, dtype=np.float32)},
                directions=self.output_specs
            )
        r1 = rank_norm(np.asarray(s1))
        r2 = rank_norm(np.asarray(s2))
        fusion = 0.5 * (r1 + r2)
        return MetricOutput(
            name=self.name,
            scores={"fusion_rankavg": fusion.astype(np.float32)},
            directions=self.output_specs
        )