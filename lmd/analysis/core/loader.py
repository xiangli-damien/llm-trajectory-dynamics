import numpy as np
from typing import Iterator, List, Tuple, Dict, Any
from .layers import apply_layer_selection
from .types import LayerSpec

class StateLoader:
    def __init__(self, arrays: Dict[str, Any], indices: np.ndarray, layer_spec: LayerSpec):
        self.arrays = arrays
        self.indices = indices
        self.layer_spec = layer_spec
    
    def load_states(self, mode: str) -> np.ndarray:
        if mode == 'mean':
            arr = self.arrays.get("mean_answer_hs")
            if arr is None:
                raise KeyError("mean_answer_hs not found")
        elif mode == 'prompt_last':
            arr = self.arrays.get("prompt_last_hs")
            if arr is None:
                raise KeyError("prompt_last_hs not found")
        else:
            raise ValueError("mode must be 'mean' or 'prompt_last'")
        
        X = arr.oindex[self.indices].astype(np.float32)
        return apply_layer_selection(X, self.layer_spec)

class TokenStreamer:
    def __init__(self, arrays: Dict[str, Any], indices: np.ndarray, 
                 layer_spec: LayerSpec, batch_size: int = 128):
        self.arrays = arrays
        self.indices = indices
        self.layer_spec = layer_spec
        self.batch_size = batch_size
        
        if arrays.get("answer_tok_values") is None or arrays.get("answer_tok_ptr") is None:
            raise KeyError("per-token arrays not available")
        
        self.tok_vals = arrays["answer_tok_values"]
        self.ptr = arrays["answer_tok_ptr"].astype(np.int64)
        
        if np.any(self.ptr[:-1] > self.ptr[1:]):
            raise ValueError("answer_tok_ptr is not monotonically increasing")
    
    def stream(self) -> Iterator[Tuple[np.ndarray, List[np.ndarray]]]:
        for s in range(0, len(self.indices), self.batch_size):
            batch = self.indices[s:s+self.batch_size]
            seqs = []
            
            for ridx in batch:
                if ridx < 0 or ridx >= len(self.ptr) - 1:
                    raise IndexError(f"Sample index {ridx} out of bounds")
                
                a, b = int(self.ptr[ridx]), int(self.ptr[ridx+1])
                
                if b <= a:
                    seqs.append(np.empty((0, 0, 0), np.float32))
                else:
                    data = self.tok_vals.oindex[a:b].astype(np.float32, copy=False)
                    data = apply_layer_selection(data, self.layer_spec)
                    seqs.append(data)
            
            yield batch, seqs