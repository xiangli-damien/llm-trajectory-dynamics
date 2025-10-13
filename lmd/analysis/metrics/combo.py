import numpy as np
from typing import Dict, List
from ..core.metrics_base import MetricBase
from ..core.types import MetricDirection, MetricOutput
from ..core.utils import rank_normalize
from . import ndr, rownull

class ComboNDRRowSimple(MetricBase):
    def __init__(self):
        super().__init__()
    
    @property
    def name(self) -> str:
        return "combo_ndr_row_simple"
    
    @property
    def requires_lm_head(self) -> bool:
        return True
    
    @property
    def supported_modes(self) -> List[str]:
        return ["state"]
    
    @property
    def output_specs(self) -> Dict[str, MetricDirection]:
        return {'combo_ndr_row_simple': MetricDirection.HIGHER_BETTER}
    
    @property
    def dependencies(self) -> List[str]:
        return ["ndr", "rownull"]
    
    def _to_higher_better(self, key: str, val: np.ndarray, 
                          ndr_specs: Dict, row_specs: Dict) -> np.ndarray:
        if key in ndr_specs and ndr_specs[key] == MetricDirection.LOWER_BETTER:
            return -val
        if key in row_specs and row_specs[key] == MetricDirection.LOWER_BETTER:
            return -val
        return val
    
    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        ndr_state = ctx.get_metric_state('ndr') or {}
        row_state = ctx.get_metric_state('rownull') or {}
        
        ndr_specs = ndr.NDR().output_specs
        row_specs = rownull.RowNull().output_specs
        
        N = len(states)
        
        ndr_keys = ['ndr_lognorm_slope', 'ndr', 'ndr_last2_gap']
        s_ndr = None
        for k in ndr_keys:
            if k in ndr_state:
                s = ndr_state[k]
                s_ndr = self._to_higher_better(k, s, ndr_specs, row_specs)
                break
        if s_ndr is None:
            s_ndr = np.zeros(N, dtype=np.float32)
        
        if 'row_len_frac' in row_state:
            s_row = row_state['row_len_frac']
            s_row = self._to_higher_better('row_len_frac', s_row, ndr_specs, row_specs)
        elif 'tail_row_fraction' in row_state:
            s_row = row_state['tail_row_fraction']
            s_row = self._to_higher_better('tail_row_fraction', s_row, ndr_specs, row_specs)
        elif 'null_cohere_len' in row_state:
            s_row = -row_state['null_cohere_len']
        else:
            s_row = np.zeros(N, dtype=np.float32)
        
        r1 = rank_normalize(s_ndr)
        r2 = rank_normalize(s_row)
        
        combo = np.sqrt(np.clip(r1, 0, 1) * np.clip(r2, 0, 1)).astype(np.float32)
        
        return MetricOutput(
            name=self.name,
            scores={'combo_ndr_row_simple': combo},
            directions=self.output_specs
        )

class ComboNDRRow3Way(MetricBase):
    def __init__(self):
        super().__init__()
    
    @property
    def name(self) -> str:
        return "combo_ndr_row_3way"
    
    @property
    def requires_lm_head(self) -> bool:
        return True
    
    @property
    def supported_modes(self) -> List[str]:
        return ["state"]
    
    @property
    def output_specs(self) -> Dict[str, MetricDirection]:
        return {'combo_ndr_row_3way': MetricDirection.HIGHER_BETTER}
    
    @property
    def dependencies(self) -> List[str]:
        return ["ndr", "rownull"]
    
    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        ndr_state = ctx.get_metric_state('ndr') or {}
        row_state = ctx.get_metric_state('rownull') or {}
        N = len(states)
        
        s_ndr = None
        for k in ['ndr_lognorm_slope', 'ndr', 'ndr_last2_gap']:
            if k in ndr_state:
                s_ndr = ndr_state[k]
                break
        if s_ndr is None:
            s_ndr = np.zeros(N, dtype=np.float32)
        
        s_row = None
        for k in ['row_len_frac', 'tail_row_fraction']:
            if k in row_state:
                s_row = row_state[k]
                break
        if s_row is None:
            s_row = np.zeros(N, dtype=np.float32)
        
        if 'tail_null_fraction' in row_state:
            s_null_inv = 1.0 - row_state['tail_null_fraction']
        elif 'null_len_frac' in row_state:
            s_null_inv = 1.0 - row_state['null_len_frac']
        else:
            s_null_inv = np.ones(N, dtype=np.float32) * 0.5
        
        r1 = rank_normalize(s_ndr)
        r2 = rank_normalize(s_row)
        r3 = rank_normalize(s_null_inv)
        
        combo = np.power(np.clip(r1, 0, 1) * np.clip(r2, 0, 1) * np.clip(r3, 0, 1), 1/3).astype(np.float32)
        
        return MetricOutput(
            name=self.name,
            scores={'combo_ndr_row_3way': combo},
            directions=self.output_specs
        )

class ComboStableV2(MetricBase):
    def __init__(self, alpha: float = 0.35):
        super().__init__(alpha=alpha)
        self.alpha = alpha
    
    @property
    def name(self) -> str:
        return "combo_stable_v2"
    
    @property
    def requires_lm_head(self) -> bool:
        return True
    
    @property
    def supported_modes(self) -> List[str]:
        return ["state"]
    
    @property
    def output_specs(self) -> Dict[str, MetricDirection]:
        return {'combo_stable_v2': MetricDirection.HIGHER_BETTER}
    
    @property
    def dependencies(self) -> List[str]:
        return ["ndr", "rownull"]
    
    def compute_state(self, ctx, states: np.ndarray) -> MetricOutput:
        ndr_state = ctx.get_metric_state('ndr') or {}
        row_state = ctx.get_metric_state('rownull') or {}
        N = len(states)
        
        ndr_core = None
        for k in ['ndr_lognorm_slope', 'ndr', 'ndr_last2_gap']:
            if k in ndr_state:
                ndr_core = ndr_state[k]
                break
        if ndr_core is None:
            ndr_core = np.zeros(N, dtype=np.float32)
        
        row_core = None
        for k in ['row_len_frac', 'tail_row_fraction']:
            if k in row_state:
                row_core = row_state[k]
                break
        if row_core is None:
            row_core = np.zeros(N, dtype=np.float32)
        
        ndr_edge = ndr_state.get('ndr_last2_gap', ndr_core)
        
        if 'null_cohere_len' in row_state:
            row_null_supp = 1.0 - row_state['null_cohere_len']
        elif 'null_len_frac' in row_state:
            row_null_supp = 1.0 - row_state['null_len_frac']
        else:
            row_null_supp = np.ones(N, dtype=np.float32) * 0.5
        
        base = np.sqrt(rank_normalize(ndr_core) * rank_normalize(row_core))
        enh = np.sqrt(rank_normalize(ndr_edge) * rank_normalize(row_null_supp))
        
        combo = ((1.0 - self.alpha) * base + self.alpha * enh).astype(np.float32)
        
        return MetricOutput(
            name=self.name,
            scores={'combo_stable_v2': combo},
            directions=self.output_specs
        )