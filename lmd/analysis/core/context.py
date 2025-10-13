from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np
from .types import RunConfig, RunData
from ..models.lm_head import LMHeadSVD

@dataclass
class RunContext:
    config: RunConfig
    data: RunData
    lm_head: Optional[LMHeadSVD]
    shared_cache: Dict[str, Any] = field(default_factory=dict)
    metric_cache: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def labels(self) -> np.ndarray:
        return self.data.labels
    
    @property
    def arrays(self) -> Dict[str, Any]:
        return self.data.arrays
    
    @property
    def sample_rows(self) -> np.ndarray:
        return self.data.row_indices
    
    @property
    def correct_mask(self) -> np.ndarray:
        return self.data.labels == 1
    
    @property
    def incorrect_mask(self) -> np.ndarray:
        return self.data.labels == 0
    
    def get_shared_cache(self, key: str) -> Any:
        return self.shared_cache.get(key)
    
    def set_shared_cache(self, key: str, value: Any):
        self.shared_cache[key] = value
    
    def get_metric_state(self, metric_name: str) -> Any:
        return self.metric_cache.get(f"{metric_name}_state")
    
    def set_metric_state(self, metric_name: str, state: Any):
        self.metric_cache[f"{metric_name}_state"] = state