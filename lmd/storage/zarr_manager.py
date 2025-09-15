"""Zarr storage management for hidden states and metadata."""

import zarr
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from numcodecs import Blosc

from ..utils.types import ModelMetadata, HiddenStateSpec


class ZarrManager:
    """Manages Zarr storage for hidden states and metadata."""
    
    def __init__(self, 
                 storage_path: Path,
                 model_metadata: ModelMetadata,
                 hidden_state_spec: HiddenStateSpec,
                 n_samples_estimate: int):
        """Initialize Zarr manager."""
        self.storage_path = Path(storage_path)
        self.model_metadata = model_metadata
        self.hidden_state_spec = hidden_state_spec
        self.n_samples_estimate = n_samples_estimate
        
        self._store = None
        self._arrays = {}
        self._current_ptr = 0
        
        self._initialize_storage()
    
    def _initialize_storage(self) -> None:
        """Initialize Zarr storage structure."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._store = zarr.open_group(str(self.storage_path), mode="a")
        
        # Set up compressor
        compressor = Blosc(cname="zstd", clevel=5)
        
        L = self.model_metadata.n_layers  # Already includes embedding
        H = self.model_metadata.hidden_dim
        
        # Create or open arrays based on spec
        if self.hidden_state_spec.need_mean:
            self._arrays["mean_answer_hs"] = self._store.require_dataset(
                "mean_answer_hs",
                shape=(self.n_samples_estimate, L, H),
                chunks=(min(1024, self.n_samples_estimate), L, H),
                dtype="float32",
                compressor=compressor
            )
        
        if self.hidden_state_spec.need_prompt_last:
            self._arrays["prompt_last_hs"] = self._store.require_dataset(
                "prompt_last_hs",
                shape=(self.n_samples_estimate, L, H),
                chunks=(min(1024, self.n_samples_estimate), L, H),
                dtype="float32",
                compressor=compressor
            )
        
        # Sample tracking arrays
        self._arrays["sample_valid"] = self._store.require_dataset(
            "sample_valid",
            shape=(self.n_samples_estimate,),
            chunks=(min(4096, self.n_samples_estimate),),
            dtype="bool",
            fill_value=False
        )
        
        self._arrays["sample_id"] = self._store.require_dataset(
            "sample_id",
            shape=(self.n_samples_estimate,),
            chunks=(min(4096, self.n_samples_estimate),),
            dtype="int64",
            fill_value=-1
        )
        
        # Per-token storage if needed
        if self.hidden_state_spec.need_per_token:
            # Check if array already exists (for resume)
            if "answer_tok_values" in self._store:
                self._arrays["answer_tok_values"] = self._store["answer_tok_values"]
            else:
                self._arrays["answer_tok_values"] = self._store.require_dataset(
                    "answer_tok_values",
                    shape=(0, L, H),
                    chunks=(max(512, L * 2), L, H),
                    dtype="float16",
                    compressor=compressor
                )
            
            self._arrays["answer_tok_ptr"] = self._store.require_dataset(
                "answer_tok_ptr",
                shape=(self.n_samples_estimate + 1,),
                chunks=(min(4096, self.n_samples_estimate + 1),),
                dtype="int64",
                fill_value=0
            )
            
            # Initialize current pointer
            self._current_ptr = self._arrays["answer_tok_values"].shape[0]
    
    def get_resume_index(self) -> int:
        """Return first unfilled sample index for resume."""
        if "sample_valid" not in self._arrays:
            return 0
        valid = np.asarray(self._arrays["sample_valid"][:], dtype=bool)
        # Find first False (unwritten) sample position
        idx = np.where(~valid)[0]
        return int(idx[0]) if idx.size else int(valid.shape[0])
    
    def save_sample(self, 
                   sample_idx: int,
                   hidden_state_data,
                   metadata: Dict[str, Any]) -> None:
        """Save a single sample's hidden state data."""
        # Save mean states
        if self.hidden_state_spec.need_mean and "mean_answer_hs" in self._arrays:
            self._arrays["mean_answer_hs"][sample_idx] = hidden_state_data.mean_states
        
        # Save prompt last states
        if self.hidden_state_spec.need_prompt_last and "prompt_last_hs" in self._arrays:
            self._arrays["prompt_last_hs"][sample_idx] = hidden_state_data.prompt_final_state
        
        # Save per-token states
        if (self.hidden_state_spec.need_per_token and 
            "answer_tok_values" in self._arrays and 
            hidden_state_data.per_token_states is not None):
            
            # Update pointer
            self._arrays["answer_tok_ptr"][sample_idx] = self._current_ptr
            
            # Resize and append data
            current_size = self._arrays["answer_tok_values"].shape[0]
            new_size = current_size + hidden_state_data.answer_token_count
            L = self.model_metadata.n_layers
            H = self.model_metadata.hidden_dim
            self._arrays["answer_tok_values"].resize((new_size, L, H))
            self._arrays["answer_tok_values"][current_size:new_size] = hidden_state_data.per_token_states
            
            self._current_ptr = new_size
            self._arrays["answer_tok_ptr"][sample_idx + 1] = self._current_ptr
        elif self.hidden_state_spec.need_per_token and "answer_tok_ptr" in self._arrays:
            # No per-token data for this sample, just update pointers
            self._arrays["answer_tok_ptr"][sample_idx] = self._current_ptr
            if sample_idx + 1 < self._arrays["answer_tok_ptr"].shape[0]:
                self._arrays["answer_tok_ptr"][sample_idx + 1] = self._current_ptr
        
        # Mark sample as valid
        self._arrays["sample_valid"][sample_idx] = True
        self._arrays["sample_id"][sample_idx] = metadata.get("sample_id", sample_idx)
    
    def mark_empty(self, sample_idx: int, sample_id: int) -> None:
        """Mark a sample as empty to maintain pointer alignment."""
        # Update pointers to maintain alignment
        if "answer_tok_ptr" in self._arrays:
            self._arrays["answer_tok_ptr"][sample_idx] = self._current_ptr
            if sample_idx + 1 < self._arrays["answer_tok_ptr"].shape[0]:
                self._arrays["answer_tok_ptr"][sample_idx + 1] = self._current_ptr
        
        # Mark as invalid
        self._arrays["sample_valid"][sample_idx] = False
        self._arrays["sample_id"][sample_idx] = sample_id
    
    def finalize(self) -> None:
        """Finalize storage operations."""
        if "answer_tok_ptr" in self._arrays:
            # Ensure final pointer is written
            self._arrays["answer_tok_ptr"][-1] = self._current_ptr
        
        # Consolidate metadata
        zarr.consolidate_metadata(self._store.store)