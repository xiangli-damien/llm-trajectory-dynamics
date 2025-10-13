from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np
from .types import MetricDirection, MetricOutput

class MetricBase(ABC):
    def __init__(self, **kwargs):
        self.params = kwargs
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def requires_lm_head(self) -> bool:
        pass
    
    @property
    @abstractmethod
    def supported_modes(self) -> List[str]:
        pass
    
    @property
    @abstractmethod
    def output_specs(self) -> Dict[str, MetricDirection]:
        pass
    
    @property
    def dependencies(self) -> List[str]:
        return []
    
    @abstractmethod
    def compute_state(self, ctx: 'RunContext', states: np.ndarray) -> MetricOutput:
        pass
    
    def compute_token(self, ctx: 'RunContext', sequences: List[np.ndarray], 
                     agg_fn) -> MetricOutput:
        raise NotImplementedError(f"{self.name} does not support token mode")
    
    def validate_params(self) -> bool:
        return True