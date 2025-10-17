import numpy as np
from typing import Dict, List
from ..core.metrics_base import MetricBase
from ..core.types import MetricDirection, MetricOutput
from ..core.utils import rank_normalize

class DynamicsCombo(MetricBase):
    def __init__(self, use_rownull: bool = True):
        super().__init__(use_rownull=use_rownull)
        self.use_rownull = bool(use_rownull)

    @property
    def name(self) -> str:
        return "din_combo"

    @property
    def requires_lm_head(self) -> bool:
        return False

    @property
    def supported_modes(self) -> List[str]:
        return ["state"]

    @property
    def output_specs(self) -> Dict[str, MetricDirection]:
        return {"din_combo": MetricDirection.HIGHER_BETTER}

    @property
    def dependencies(self) -> List[str]:
        return ["nrleak", "dac", "als"]

    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        N = states.shape[0]
        
        nrleak_state = ctx.get_metric_state('nrleak') or {}
        dac_state = ctx.get_metric_state('dac') or {}
        als_state = ctx.get_metric_state('als') or {}
        rownull_state = ctx.get_metric_state('rownull') or {}

        components = []
        
        if 'nrleak_leak_mean' in nrleak_state:
            components.append(rank_normalize(-np.asarray(nrleak_state['nrleak_leak_mean'], dtype=np.float32)))
        else:
            components.append(np.zeros(N, dtype=np.float32))
            
        if 'nrleak_qtc' in nrleak_state:
            components.append(rank_normalize(-np.asarray(nrleak_state['nrleak_qtc'], dtype=np.float32)))
        else:
            components.append(np.zeros(N, dtype=np.float32))

        if 'dac_cohere' in dac_state:
            components.append(rank_normalize(np.asarray(dac_state['dac_cohere'], dtype=np.float32)))
        else:
            components.append(np.zeros(N, dtype=np.float32))
            
        if 'dac_fliprate' in dac_state:
            components.append(rank_normalize(-np.asarray(dac_state['dac_fliprate'], dtype=np.float32)))
        else:
            components.append(np.zeros(N, dtype=np.float32))

        if 'als_stability' in als_state:
            components.append(rank_normalize(np.asarray(als_state['als_stability'], dtype=np.float32)))
        else:
            components.append(np.zeros(N, dtype=np.float32))

        if self.use_rownull and ('row_len_frac' in rownull_state):
            components.append(rank_normalize(np.asarray(rownull_state['row_len_frac'], dtype=np.float32)))

        X = np.stack(components, axis=1)
        din_score = np.mean(X, axis=1).astype(np.float32)

        return MetricOutput(
            name=self.name,
            scores={"din_combo": din_score},
            directions=self.output_specs,
            cache_state={"din_combo": din_score}
        )