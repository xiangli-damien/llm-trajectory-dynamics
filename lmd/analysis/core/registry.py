from typing import Dict, Type, Any, List
from .metrics_base import MetricBase
from .types import MetricConfig, MetricDirection

class MetricRegistry:
    def __init__(self):
        self._metrics: Dict[str, Type[MetricBase]] = {}
        self._instances: Dict[str, MetricBase] = {}
    
    def register(self, name: str, metric_class: Type[MetricBase]):
        if name in self._metrics:
            raise KeyError(f"Metric {name} already registered")
        self._metrics[name] = metric_class
    
    def create_instance(self, config: MetricConfig) -> MetricBase:
        if config.name not in self._metrics:
            raise KeyError(f"Metric {config.name} not found in registry")
        
        metric_class = self._metrics[config.name]
        instance = metric_class(**config.params)
        
        if not instance.validate_params():
            raise ValueError(f"Invalid parameters for metric {config.name}: {config.params}")
        
        return instance
    
    def get_instance(self, name: str, params: Dict[str, Any] = None) -> MetricBase:
        key = f"{name}_{hash(frozenset((params or {}).items()))}"
        
        if key not in self._instances:
            config = MetricConfig(name=name, params=params or {})
            self._instances[key] = self.create_instance(config)
        
        return self._instances[key]
    
    def list_metrics(self) -> List[str]:
        return list(self._metrics.keys())
    
    def get_metric_info(self, name: str) -> Dict[str, Any]:
        if name not in self._metrics:
            raise KeyError(f"Metric {name} not found")
        
        metric_class = self._metrics[name]
        temp_instance = metric_class()
        
        return {
            'name': name,
            'requires_lm_head': temp_instance.requires_lm_head,
            'supported_modes': temp_instance.supported_modes,
            'output_specs': temp_instance.output_specs,
            'dependencies': temp_instance.dependencies
        }