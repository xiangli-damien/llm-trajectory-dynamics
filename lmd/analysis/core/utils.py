import time
import psutil
import warnings
import numpy as np
from contextlib import contextmanager
from typing import Optional

@contextmanager
def Timer(name: str = "Operation"):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{name} took {elapsed:.3f} seconds")

class MemoryLimiter:
    def __init__(self, max_gb: float = 8.0):
        self.max_bytes = max_gb * (1 << 30)
    
    def check(self):
        process = psutil.Process()
        mem = process.memory_info().rss
        if mem > self.max_bytes:
            warnings.warn(f"Memory usage ({mem/(1<<30):.2f} GB) exceeds limit ({self.max_bytes/(1<<30):.2f} GB)")
            return False
        return True

class ThreadLimiter:
    def __init__(self, max_threads: Optional[int] = None):
        self.max_threads = max_threads or psutil.cpu_count()
    
    def get_n_jobs(self, requested: int = -1) -> int:
        if requested == -1:
            return self.max_threads
        return min(requested, self.max_threads)

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

def auto_tail_window_from_dr(dr: np.ndarray, cover: float = 0.8, min_w: int = 2) -> int:
    imp = np.median(np.abs(dr), axis=0)
    tot = np.sum(imp) + 1e-12
    acc = 0.0
    w = 1
    for i in range(len(imp)-1, -1, -1):
        acc += imp[i]
        w += 1
        if acc / tot >= cover:
            break
    return max(min_w, w)