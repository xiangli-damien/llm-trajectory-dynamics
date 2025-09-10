"""Hidden state extraction and processing utilities."""

import torch
import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

from ..utils.types import HiddenStateSpec


@dataclass
class HiddenStateData:
    """Container for extracted hidden state data."""
    per_token_states: Optional[np.ndarray]  # (T, L, H)
    mean_states: np.ndarray                 # (L, H)
    prompt_final_state: np.ndarray          # (L, H)
    answer_token_count: int


class HiddenStateExtractor:
    """Extracts and processes hidden states from model outputs."""
    
    def __init__(self, spec: HiddenStateSpec, per_token_dtype: str = "float16", stat_dtype: str = "float32"):
        """Initialize hidden state extractor."""
        self.spec = spec
        self.per_token_dtype = per_token_dtype
        self.stat_dtype = stat_dtype
    
    def extract_from_generation(self, 
                              generation_result,
                              prompt_hidden_states: List[torch.Tensor]) -> HiddenStateData:
        """Extract hidden states from generation result."""
        # Extract per-token states for answer tokens
        expected_layers = len(prompt_hidden_states)  # Align with prompt (excluding embedding)
        per_token_states = None
        if self.spec.need_per_token and generation_result.generated_length > 0:
            per_token_states = self._extract_per_token_states(
                generation_result.hidden_states, expected_layers=expected_layers
            )
        
        # Compute mean states
        mean_states = self._compute_mean_states(
            per_token_states, generation_result.generated_length, prompt_hidden_states
        )
        
        # Extract prompt final state
        prompt_final_state = self._extract_prompt_final_state(prompt_hidden_states)
        
        return HiddenStateData(
            per_token_states=per_token_states,
            mean_states=mean_states,
            prompt_final_state=prompt_final_state,
            answer_token_count=generation_result.generated_length
        )
    
    def _extract_per_token_states(self, hidden_states: List[Tuple[torch.Tensor, ...]], expected_layers: int) -> np.ndarray:
        """Extract per-token hidden states from generation steps."""
        if not hidden_states:
            # Return None to trigger upper-level zero vector fallback, avoid shape crash
            return None
        
        per_token_data = []
        num_steps = len(hidden_states)
        target_dtype_torch = torch.float16 if self.per_token_dtype == "float16" else torch.float32
        target_dtype_np = np.float16 if self.per_token_dtype == "float16" else np.float32
        
        for step_idx in range(num_steps):
            layers = list(hidden_states[step_idx])
            if len(layers) == expected_layers + 1:
                layers = layers[1:]
            elif len(layers) != expected_layers and len(layers) > 1:
                try:
                    if layers[0].shape[-1] == layers[1].shape[-1]:
                        layers = layers[1:]
                except Exception:
                    pass
            
            layer_vectors = []
            for layer_output in layers:
                if layer_output.dim() == 3:
                    vector = layer_output[0, 0, :].to(target_dtype_torch).cpu().numpy().astype(target_dtype_np)
                else:
                    vector = layer_output[0, :].to(target_dtype_torch).cpu().numpy().astype(target_dtype_np)
                layer_vectors.append(vector)
            
            per_token_data.append(np.stack(layer_vectors, axis=0))
        
        return np.stack(per_token_data, axis=0).astype(target_dtype_np)
    
    def _compute_mean_states(self, 
                           per_token_states: Optional[np.ndarray],
                           generated_length: int,
                           prompt_hidden_states: List[torch.Tensor]) -> np.ndarray:
        """Compute mean hidden states across answer tokens."""
        target_dtype = np.float32 if self.stat_dtype == "float32" else np.float16
        
        if generated_length == 0:
            num_layers = len(prompt_hidden_states)
            hidden_dim = prompt_hidden_states[0].shape[-1]
            return np.zeros((num_layers, hidden_dim), dtype=target_dtype)
        
        if per_token_states is not None:
            return per_token_states.mean(axis=0, dtype=target_dtype)
        else:
            num_layers = len(prompt_hidden_states)
            hidden_dim = prompt_hidden_states[0].shape[-1]
            return np.zeros((num_layers, hidden_dim), dtype=target_dtype)
    
    def _extract_prompt_final_state(self, prompt_hidden_states: List[torch.Tensor]) -> np.ndarray:
        """Extract hidden state of the final prompt token."""
        layer_vectors = []
        target_dtype_torch = torch.float32 if self.stat_dtype == "float32" else torch.float16
        target_dtype_np = np.float32 if self.stat_dtype == "float32" else np.float16
        
        for layer_output in prompt_hidden_states:
            vector = layer_output[0, -1, :].to(target_dtype_torch).cpu().numpy().astype(target_dtype_np)
            layer_vectors.append(vector)
        
        return np.stack(layer_vectors, axis=0).astype(target_dtype_np)
    
    def extract_from_full_forward(self, 
                                full_hidden_states: List[torch.Tensor],
                                answer_span: Tuple[int, int]) -> np.ndarray:
        """Extract per-token states from full forward pass (legacy method)."""
        start_idx, end_idx = answer_span
        layer_vectors = []
        target_dtype_np = np.float16 if self.per_token_dtype == "float16" else np.float32
        
        for layer_output in full_hidden_states:
            answer_tokens = layer_output[0, start_idx:end_idx, :].to(
                torch.float16 if self.per_token_dtype == "float16" else torch.float32
            ).cpu().numpy().astype(target_dtype_np)
            layer_vectors.append(answer_tokens)
        
        result = np.stack(layer_vectors, axis=1)
        return result.astype(target_dtype_np)
    
    @staticmethod
    def locate_answer_span(input_length: int, generated_length: int) -> Tuple[int, int]:
        """Locate answer token span in full sequence."""
        return (input_length, input_length + generated_length)
