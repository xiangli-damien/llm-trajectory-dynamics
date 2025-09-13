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
        """Extract per-token hidden states from generation steps."""
        if not hidden_states or generated_length == 0:
            return None
        
        # Get layer count from first step
        num_layers = len(hidden_states[0])
        
        # Collect all token states
        per_token_data = []
        for step_idx in range(generated_length):
            step_states = hidden_states[step_idx]
            
            # Extract hidden state for current token at each layer
            layer_vectors = []
            for layer_idx in range(num_layers):
                # Get the last token's hidden state (newly generated token)
                hs = step_states[layer_idx]
                if hs.dim() == 3:
                    vector = hs[0, -1, :].float().cpu().numpy()
                elif hs.dim() == 2:
                    vector = hs[-1, :].float().cpu().numpy()
                else:
                    vector = hs[0, :].float().cpu().numpy()
                layer_vectors.append(vector)
            
            per_token_data.append(np.stack(layer_vectors, axis=0))
        
        return np.stack(per_token_data, axis=0).astype(np.float16)
    
    def _compute_mean_states(self, hidden_states: List, generated_length: int) -> np.ndarray:
        """
        Compute mean hidden states across answer tokens.
        Aligns with CoE: includes ALL generated tokens (including EOS if present).
        """
        if not hidden_states or generated_length == 0:
            # Return zeros with correct shape
            return self._zeros_lh(np.float32)
        
        # Get layer count
        num_layers = len(hidden_states[0])
        
        # Verify layer count matches metadata
        if num_layers != self.model_meta.n_layers:
            print(f"Warning: Layer mismatch - runtime={num_layers}, metadata={self.model_meta.n_layers}")
            # Use minimum to avoid index errors
            num_layers = min(num_layers, self.model_meta.n_layers)
        
        # Collect states from all generated tokens
        layer_means = []
        for layer_idx in range(num_layers):
            token_states = []
            
            # Collect states from each generation step
            for step_idx in range(generated_length):
                step_states = hidden_states[step_idx]
                hs = step_states[layer_idx]
                
                # Extract last token's state (newly generated) with robust shape handling
                if hs.dim() == 3:
                    vector = hs[0, -1, :].float().cpu().numpy()
                elif hs.dim() == 2:
                    vector = hs[-1, :].float().cpu().numpy()
                else:
                    vector = hs[0, :].float().cpu().numpy()
                token_states.append(vector)
            
            # Compute mean for this layer
            layer_mean = np.mean(token_states, axis=0, dtype=np.float32)
            layer_means.append(layer_mean)
        
        # Pad with zeros if fewer layers than expected
        while len(layer_means) < self.model_meta.n_layers:
            layer_means.append(np.zeros(self.model_meta.hidden_dim, dtype=np.float32))
        
        return np.stack(layer_means, axis=0).astype(np.float32)
    
    def _extract_prompt_final_state(self, prompt_hidden_states: List[torch.Tensor]) -> np.ndarray:
        """Extract hidden state of the final prompt token."""
        if not prompt_hidden_states:
            # Return zeros with correct shape
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