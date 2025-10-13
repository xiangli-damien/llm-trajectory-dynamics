import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import warnings

try:
    from safetensors.torch import load_file as safe_load_torch
except ImportError:
    safe_load_torch = None

def _choose_rank_by_variance(S: np.ndarray, var_ratio: Optional[float], 
                            requested_rank: Optional[int], hidden_dim: int) -> Tuple[int, float]:
    total = float(np.sum(S.astype(np.float64)**2))
    if total <= 0:
        return hidden_dim, 0.0
    
    if requested_rank is not None:
        r = min(int(requested_rank), len(S))
        exp = float(np.sum(S[:r].astype(np.float64)**2))/total
        return r, exp
    
    if var_ratio is None:
        target = 0.95
    else:
        target = float(var_ratio)
    
    csum = np.cumsum((S**2).astype(np.float64))
    r = int(np.searchsorted(csum, target * total) + 1)
    r = max(1, min(r, len(S)))
    exp = float(csum[r-1]/total) if r > 0 else 0.0
    return r, exp

@dataclass
class LMHeadSVD:
    U: np.ndarray
    S: np.ndarray
    Vh: np.ndarray
    W: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    vocab_size: int = 0
    hidden_dim: int = 0
    rank: int = 0
    explained_variance: float = 0.0
    is_exact: bool = False
    
    def compute_logits(self, h: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        input_shape = h.shape
        h_flat = h.reshape(-1, self.hidden_dim)
        
        if self.W is not None and self.is_exact:
            Z = h_flat @ self.W.T
        else:
            y = (h_flat @ self.Vh.T) * self.S
            Z = y @ self.U.T
        
        if self.bias is not None:
            Z = Z + self.bias
        
        Z = Z / float(temperature)
        
        if len(input_shape) == 1:
            return Z[0]
        elif len(input_shape) == 2:
            return Z.reshape(input_shape[0], -1)
        else:
            return Z.reshape(*input_shape[:-1], -1)
    
    def project_hidden(self, h: np.ndarray) -> np.ndarray:
        r = self.rank
        H = self.hidden_dim
        h_flat = h.reshape(-1, H)
        y = h_flat @ self.Vh[:r, :].T
        return y.reshape(*h.shape[:-1], r).astype(h.dtype, copy=False)
    
    def compute_logits_selected_from_y(self, y: np.ndarray, idx: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        r = self.rank
        y_flat = y.reshape(-1, r)
        US = self.U[idx, :r]
        z = (y_flat * self.S[:r]) @ US.T
        if self.bias is not None:
            z = z + self.bias[idx]
        z = z / float(temperature)
        return z.reshape(*y.shape[:-1], len(idx))
    
    def compute_logits_selected(self, h: np.ndarray, idx: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        y = self.project_hidden(h)
        return self.compute_logits_selected_from_y(y, idx, temperature)
    
    def readout_projection(self, rank: Optional[int] = None, 
                          var_ratio: Optional[float] = None) -> np.ndarray:
        if rank is None and var_ratio is None:
            r = self.rank
        else:
            r, _ = _choose_rank_by_variance(self.S, var_ratio, rank, self.hidden_dim)
        
        r = min(r, self.rank, self.Vh.shape[0])
        V = self.Vh[:r, :].T.astype(np.float32)
        Q, _ = np.linalg.qr(V)
        return Q[:, :r]

def load_lm_head_weight_only(model_path: Union[str, Path]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    mp = Path(model_path)
    
    W = None
    bias = None
    
    if safe_load_torch is not None:
        for f in sorted(mp.glob("*.safetensors")):
            try:
                tensors = safe_load_torch(str(f), device="cpu")
                for k in ["lm_head.weight", "model.lm_head.weight", "output.weight"]:
                    if k in tensors:
                        tensor = tensors[k]
                        if hasattr(tensor, 'float'):
                            tensor = tensor.float()
                        W = tensor.cpu().numpy().astype(np.float32, copy=False)
                        
                        bias_key = k.replace('.weight', '.bias')
                        if bias_key in tensors:
                            bias_tensor = tensors[bias_key]
                            if hasattr(bias_tensor, 'float'):
                                bias_tensor = bias_tensor.float()
                            bias = bias_tensor.cpu().numpy().astype(np.float32, copy=False)
                        
                        return W, bias
            except:
                continue
    
    for f in sorted(mp.glob("*.bin")):
        try:
            sd = torch.load(f, map_location="cpu", weights_only=True)
            for k in ["lm_head.weight", "model.lm_head.weight", "output.weight"]:
                if k in sd:
                    tensor = sd[k]
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.float()
                    W = tensor.cpu().numpy().astype(np.float32, copy=False)
                    
                    bias_key = k.replace('.weight', '.bias')
                    if bias_key in sd:
                        bias_tensor = sd[bias_key]
                        if bias_tensor.dtype == torch.bfloat16:
                            bias_tensor = bias_tensor.float()
                        bias = bias_tensor.cpu().numpy().astype(np.float32, copy=False)
                    
                    return W, bias
        except:
            continue
    
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            str(mp),
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        lm_head = model.get_output_embeddings()
        W = lm_head.weight.detach().cpu().numpy().astype(np.float32, copy=False)
        bias = None
        if hasattr(lm_head, 'bias') and lm_head.bias is not None:
            bias = lm_head.bias.detach().cpu().numpy().astype(np.float32, copy=False)
        del model
        torch.cuda.empty_cache()
        return W, bias
    except Exception as e:
        raise ValueError(f"Could not load LM head weights from {mp}: {e}")

def load_lm_head(model_path: Union[str, Path], 
                 rank: Optional[int] = None,
                 exact_mode: bool = False,
                 var_ratio: Optional[float] = None,
                 dtype: str = 'float32') -> LMHeadSVD:
    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f"Model path not found: {model_path}")
    
    W, bias = load_lm_head_weight_only(model_path)
    vocab_size, hidden_dim = W.shape
    
    if rank is None and not exact_mode and var_ratio is None:
        var_ratio = 0.95
    
    if hidden_dim < vocab_size // 4:
        G = W.T @ W
        eigvals, V = np.linalg.eigh(G)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        V = V[:, idx]
        
        S_full = np.sqrt(np.maximum(eigvals, 0))
        Vh_full = V.T
        
        U_full = W @ V
        for i in range(len(S_full)):
            if S_full[i] > 1e-8:
                U_full[:, i] = U_full[:, i] / S_full[i]
        
        col_norms = np.linalg.norm(U_full, axis=0) + 1e-12
        U_full = U_full / col_norms
        S_full = S_full * col_norms
    else:
        U_full, S_full, Vh_full = np.linalg.svd(W, full_matrices=False)
    
    if exact_mode:
        final_rank = hidden_dim
        U = U_full
        S = S_full
        Vh = Vh_full
        W_keep = W
        is_exact = True
    else:
        final_rank, explained = _choose_rank_by_variance(S_full, var_ratio, rank, hidden_dim)
        U = U_full[:, :final_rank]
        S = S_full[:final_rank]
        Vh = Vh_full[:final_rank, :]
        W_keep = None
        is_exact = False
    
    total_var = float(np.sum(W.astype(np.float64)**2))
    explained_var = float(np.sum(S[:final_rank].astype(np.float64)**2)) / (total_var + 1e-12)
    
    return LMHeadSVD(
        U=U.astype(dtype),
        S=S.astype(dtype),
        Vh=Vh.astype(dtype),
        W=W_keep.astype(dtype) if W_keep is not None else None,
        bias=bias.astype(dtype) if bias is not None else None,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        rank=final_rank,
        explained_variance=explained_var,
        is_exact=is_exact
    )