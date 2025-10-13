import numpy as np
from typing import Union, Optional, Dict, Any
from .types import LayerSpec

def resolve_layer_indices(total_layers: int, spec: LayerSpec) -> Union[slice, np.ndarray]:
    """Resolve layer specification to indices or slice."""
    if spec.use_layers is not None:
        indices = np.array(spec.use_layers, dtype=np.int64)
        indices = indices[indices < total_layers]
        return indices
    
    start = 0
    end = total_layers
    
    if spec.drop_embedding and total_layers > 1:
        start = 1
    
    if spec.layer_range is not None:
        range_start, range_end = spec.layer_range
        start = max(start, range_start)
        end = min(end, range_end)
    
    if spec.exclude_last_n > 0:
        end = max(start, end - spec.exclude_last_n)
    
    if start >= end:
        raise ValueError(f"Invalid layer configuration: {start}:{end} of {total_layers}")
    
    if spec.stride == 1:
        return slice(start, end)
    else:
        return np.arange(start, end, spec.stride, dtype=np.int64)

def apply_layer_selection(data: np.ndarray, spec: LayerSpec) -> np.ndarray:
    """Apply layer selection to data array."""
    if data.ndim < 2:
        return data
    
    layer_axis = 1 if data.ndim >= 3 else 0
    total_layers = data.shape[layer_axis]
    indices = resolve_layer_indices(total_layers, spec)
    
    if isinstance(indices, slice):
        if data.ndim == 3:
            return data[:, indices, :]
        else:
            return data[indices]
    else:
        if data.ndim == 3:
            return data[:, indices, :]
        else:
            return data[indices]

def compute_shared_cache(states: np.ndarray, lm_head=None, 
                        var_ratio: float = 0.95) -> Dict[str, np.ndarray]:
    """Compute shared cache for metrics."""
    N, L, H = states.shape
    cache = {}
    
    norms = np.linalg.norm(states, axis=2).astype(np.float32)
    cache['norms'] = norms
    cache['lognorms'] = np.log(np.maximum(norms, 1e-12)).astype(np.float32)
    
    if L > 1:
        cache['dlognorms'] = np.diff(cache['lognorms'], axis=1).astype(np.float32)
        dh = np.diff(states, axis=1)
        cache['dh'] = dh.astype(np.float32)
        cache['dh_energy'] = np.sum(dh * dh, axis=2).astype(np.float32)
    
    if lm_head is not None:
        Q = lm_head.readout_projection(var_ratio=var_ratio).astype(np.float32)
        cache['Q'] = Q
        
        q = np.tensordot(states, Q, axes=([2], [0])).astype(np.float32)
        cache['q'] = q
        cache['q_norm'] = np.linalg.norm(q, axis=2).astype(np.float32)
        
        if L > 1:
            dq = np.diff(q, axis=1)
            cache['dq'] = dq.astype(np.float32)
            cache['dq_norm'] = np.linalg.norm(dq, axis=2).astype(np.float32)
            cache['dr'] = np.diff(cache['q_norm'], axis=1).astype(np.float32)
    
    return cache