import numpy as np
from typing import Callable

class Aggregators:
    @staticmethod
    def get(name: str, **params) -> Callable[[np.ndarray], float]:
        if name == "median":
            return Aggregators.median
        elif name == "mean":
            return Aggregators.mean
        elif name == "quantile":
            q = params.get('q', 0.75)
            return Aggregators.quantile(q)
        elif name == "topk_mean":
            k = params.get('k', 8)
            return Aggregators.topk_mean(k)
        elif name == "max":
            return Aggregators.max
        elif name == "min":
            return Aggregators.min
        else:
            raise ValueError(f"Unknown aggregator: {name}")
    
    @staticmethod
    def median(v: np.ndarray) -> float:
        return float(np.median(v)) if v.size else 0.0
    
    @staticmethod
    def mean(v: np.ndarray) -> float:
        return float(np.mean(v)) if v.size else 0.0
    
    @staticmethod
    def max(v: np.ndarray) -> float:
        return float(np.max(v)) if v.size else 0.0
    
    @staticmethod
    def min(v: np.ndarray) -> float:
        return float(np.min(v)) if v.size else 0.0
    
    @staticmethod
    def quantile(q: float = 0.75) -> Callable[[np.ndarray], float]:
        def fn(v: np.ndarray) -> float:
            return float(np.quantile(v, q)) if v.size else 0.0
        return fn
    
    @staticmethod
    def topk_mean(k: int = 8) -> Callable[[np.ndarray], float]:
        def fn(v: np.ndarray) -> float:
            if v.size == 0:
                return 0.0
            k2 = min(max(k, 1), v.size)
            idx = np.argpartition(v, -k2)[-k2:]
            return float(np.mean(v[idx]))
        return fn