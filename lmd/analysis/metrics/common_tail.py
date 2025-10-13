import numpy as np
from typing import Optional

def rank_normalize(x: np.ndarray) -> np.ndarray:
    x = x + 1e-12 * np.random.RandomState(0).randn(*x.shape)
    xs = np.sort(x)
    idx = np.searchsorted(xs, x, side='right')
    return idx.astype(np.float32) / (len(x) + 1e-12)

def robust_zscore(x: np.ndarray, axis: Optional[int] = None, 
                 min_samples: int = 5) -> np.ndarray:
    if x.shape[0] < min_samples:
        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        return (x - mean) / (std + 1e-12)
    
    median = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - median), axis=axis, keepdims=True)
    
    if np.any(mad < 1e-10):
        std = np.std(x, axis=axis, keepdims=True)
        return (x - np.mean(x, axis=axis, keepdims=True)) / (std + 1e-12)
    
    return (x - median) / (mad + 1e-12)