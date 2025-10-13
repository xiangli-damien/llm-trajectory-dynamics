"""Hidden state extraction and processing utilities."""

import torch
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from ..utils.types import HiddenStateSpec, ModelMetadata


@dataclass
class HiddenStateData:
    """Container for extracted hidden state data."""
    per_token_states: Optional[np.ndarray]  # (T, L, H)
    mean_states: np.ndarray                 # (L, H)
    prompt_final_state: np.ndarray          # (L, H)
    answer_token_count: int


class HiddenStateExtractor:
    """Extracts and processes hidden states from model outputs."""
    
    def __init__(self, spec: HiddenStateSpec, model_meta: ModelMetadata):
        """Initialize hidden state extractor with model metadata."""
        self.spec = spec
        self.model_meta = model_meta
    
    def _zeros_lh(self, dtype=np.float32):
        """Create zero array with correct layer and hidden dimensions."""
        return np.zeros((self.model_meta.n_layers, self.model_meta.hidden_dim), dtype=dtype)
    
    def extract_from_generation(self, 
                              generation_result,
                              prompt_hidden_states: List[torch.Tensor]) -> HiddenStateData:
        """Extract hidden states from generation result"""
        hidden_states = generation_result.hidden_states
        generated_length = generation_result.generated_length
        
        # Extract per-token states if needed
        per_token_states = None
        if self.spec.need_per_token and generated_length > 0:
            per_token_states = self._extract_per_token_states(hidden_states, generated_length)
        
        # Compute mean states (CoE-aligned: mean of ALL generated tokens including EOS)
        mean_states = self._compute_mean_states(hidden_states, generated_length)
        
        # Extract prompt final state
        prompt_final_state = self._extract_prompt_final_state(prompt_hidden_states)
        
        return HiddenStateData(
            per_token_states=per_token_states,
            mean_states=mean_states,
            prompt_final_state=prompt_final_state,
            answer_token_count=generated_length
        )
    
    def _extract_per_token_states(self, hidden_states: List, generated_length: int) -> np.ndarray:
        """Extract per-token hidden states with optimized memory usage."""
        if not hidden_states or generated_length == 0:
            return None
        
        # Get dimensions
        num_layers = len(hidden_states[0])
        H = self.model_meta.hidden_dim
        
        # Pre-allocate array in float16 to save memory
        out = np.empty((generated_length, num_layers, H), dtype=np.float16)
        
        # Fill directly without intermediate lists
        for step_idx in range(generated_length):
            step_states = hidden_states[step_idx]
            for layer_idx in range(num_layers):
                hs = step_states[layer_idx]
                # Extract last token and convert to float16 directly
                if hs.dim() == 3:
                    vec = hs[0, -1, :].to(torch.float16).cpu().numpy()
                elif hs.dim() == 2:
                    vec = hs[-1, :].to(torch.float16).cpu().numpy()
                else:
                    vec = hs[0, :].to(torch.float16).cpu().numpy()
                out[step_idx, layer_idx, :] = vec
        
        return out
    
    def _compute_mean_states(self, hidden_states: List, generated_length: int) -> np.ndarray:
        """Compute mean states with online algorithm to reduce memory."""
        if not hidden_states or generated_length == 0:
            return self._zeros_lh(np.float32)
        
        num_layers = len(hidden_states[0])
        num_layers = min(num_layers, self.model_meta.n_layers)
        H = self.model_meta.hidden_dim
        
        # Initialize mean accumulator
        means = np.zeros((num_layers, H), dtype=np.float32)
        
        # Online mean calculation
        for step_idx in range(generated_length):
            step_states = hidden_states[step_idx]
            for layer_idx in range(num_layers):
                hs = step_states[layer_idx]
                # Extract vector
                if hs.dim() == 3:
                    v = hs[0, -1, :].to(torch.float32).cpu().numpy()
                elif hs.dim() == 2:
                    v = hs[-1, :].to(torch.float32).cpu().numpy()
                else:
                    v = hs[0, :].to(torch.float32).cpu().numpy()
                # Update online mean
                means[layer_idx] += (v - means[layer_idx]) / (step_idx + 1)
        
        # Pad if needed
        if means.shape[0] < self.model_meta.n_layers:
            padding = np.zeros((self.model_meta.n_layers - means.shape[0], H), dtype=np.float32)
            means = np.vstack([means, padding])
        
        return means
    
    def _extract_prompt_final_state(self, prompt_hidden_states: List[torch.Tensor]) -> np.ndarray:
        """Extract hidden state of the final prompt token."""
        if not prompt_hidden_states:
            return self._zeros_lh(np.float32)
        
        layer_vectors = []
        for layer_output in prompt_hidden_states:
            # Get last token of prompt
            vector = layer_output[0, -1, :].float().cpu().numpy()
            layer_vectors.append(vector)
        
        # Pad if needed
        while len(layer_vectors) < self.model_meta.n_layers:
            layer_vectors.append(np.zeros(self.model_meta.hidden_dim, dtype=np.float32))
        
        return np.stack(layer_vectors[:self.model_meta.n_layers], axis=0).astype(np.float32)