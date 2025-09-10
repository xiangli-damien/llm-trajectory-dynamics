"""Zarr storage management for hidden states and metadata."""

import zarr
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from numcodecs import Blosc

from ..utils.types import ModelMetadata, HiddenStateSpec


class ZarrManager:
    """Manages Zarr storage for hidden states and metadata."""
    
    def __init__(self, 
                 storage_path: Path,
                 model_metadata: ModelMetadata,
                 hidden_state_spec: HiddenStateSpec,
                 n_samples_estimate: int,
                 per_token_dtype: str = "float16",
                 mean_dtype: str = "float32",
                 prompt_dtype: str = "float32"):
        """Initialize Zarr manager."""
        self.storage_path = Path(storage_path)
        self.model_metadata = model_metadata
        self.hidden_state_spec = hidden_state_spec
        self.n_samples_estimate = n_samples_estimate
        self.per_token_dtype = per_token_dtype
        self.mean_dtype = mean_dtype
        self.prompt_dtype = prompt_dtype
        
        self._store = None
        self._arrays = {}
        self._initialized = False
        self._current_ptr = 0
        
        self._initialize_storage()
    
    def _initialize_storage(self) -> None:
        """Initialize Zarr storage structure."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._store = zarr.open_group(str(self.storage_path), mode="a")
        
        # Set up compressor
        compressor = Blosc(cname="zstd", clevel=5)
        
        if self.hidden_state_spec.need_mean:
            if "mean_answer_hs" in self._store:
                self._arrays["mean_answer_hs"] = self._store["mean_answer_hs"]
                # Align n_samples_estimate
                self.n_samples_estimate = max(self.n_samples_estimate, self._arrays["mean_answer_hs"].shape[0])
            else:
                self._arrays["mean_answer_hs"] = self._store.create_dataset(
                    "mean_answer_hs",
                    shape=(self.n_samples_estimate, self.model_metadata.n_layers, self.model_metadata.hidden_dim),
                    chunks=(min(1024, self.n_samples_estimate), self.model_metadata.n_layers, self.model_metadata.hidden_dim),
                    dtype=self.mean_dtype, compressor=compressor
                )
        
        if self.hidden_state_spec.need_prompt_last:
            if "prompt_last_hs" in self._store:
                self._arrays["prompt_last_hs"] = self._store["prompt_last_hs"]
                self.n_samples_estimate = max(self.n_samples_estimate, self._arrays["prompt_last_hs"].shape[0])
            else:
                self._arrays["prompt_last_hs"] = self._store.create_dataset(
                    "prompt_last_hs",
                    shape=(self.n_samples_estimate, self.model_metadata.n_layers, self.model_metadata.hidden_dim),
                    chunks=(min(1024, self.n_samples_estimate), self.model_metadata.n_layers, self.model_metadata.hidden_dim),
                    dtype=self.prompt_dtype, compressor=compressor
                )
        
        if self.hidden_state_spec.need_per_token:
            if "answer_tok_values" in self._store:
                self._arrays["answer_tok_values"] = self._store["answer_tok_values"]
            else:
                self._arrays["answer_tok_values"] = self._store.create_dataset(
                    "answer_tok_values",
                    shape=(0, self.model_metadata.n_layers, self.model_metadata.hidden_dim),
                    chunks=(max(512, self.model_metadata.n_layers * 2), self.model_metadata.n_layers, self.model_metadata.hidden_dim),
                    dtype=self.per_token_dtype, compressor=compressor
                )
            
            if "answer_tok_ptr" in self._store:
                self._arrays["answer_tok_ptr"] = self._store["answer_tok_ptr"]
                # Ensure n_samples_estimate aligns with ptr length-1
                self.n_samples_estimate = max(self.n_samples_estimate, self._arrays["answer_tok_ptr"].shape[0] - 1)
            else:
                self._arrays["answer_tok_ptr"] = self._store.create_dataset(
                    "answer_tok_ptr",
                    shape=(self.n_samples_estimate + 1,),
                    chunks=(min(4096, self.n_samples_estimate + 1),),
                    dtype="int64"
                )
            
            if "sample_valid" in self._store and "sample_id" in self._store:
                self._arrays["sample_valid"] = self._store["sample_valid"]
                self._arrays["sample_id"] = self._store["sample_id"]
                self.n_samples_estimate = max(self.n_samples_estimate, self._arrays["sample_valid"].shape[0])
            else:
                self._arrays["sample_valid"] = self._store.create_dataset(
                    "sample_valid",
                    shape=(self.n_samples_estimate,),
                    chunks=(min(4096, self.n_samples_estimate),),
                    dtype="bool"
                )
                self._arrays["sample_valid"][:] = False
                self._arrays["sample_id"] = self._store.create_dataset(
                    "sample_id",
                    shape=(self.n_samples_estimate,),
                    chunks=(min(4096, self.n_samples_estimate),),
                    dtype="int64"
                )
                self._arrays["sample_id"][:] = -1
            
            # Ensure row-wise capacity grows when resuming with larger dataset
            L, H = self.model_metadata.n_layers, self.model_metadata.hidden_dim
            existing_n = int(self._arrays["sample_valid"].shape[0])
            target_n = max(existing_n, int(self.n_samples_estimate))
            if target_n > existing_n:
                # Expand mean/prompt arrays if they exist
                if "mean_answer_hs" in self._arrays and self._arrays["mean_answer_hs"].shape[0] < target_n:
                    self._arrays["mean_answer_hs"].resize((target_n, L, H))
                if "prompt_last_hs" in self._arrays and self._arrays["prompt_last_hs"].shape[0] < target_n:
                    self._arrays["prompt_last_hs"].resize((target_n, L, H))
                # Expand sample_valid / sample_id
                self._arrays["sample_valid"].resize((target_n,))
                self._arrays["sample_valid"][existing_n:target_n] = False
                self._arrays["sample_id"].resize((target_n,))
                self._arrays["sample_id"][existing_n:target_n] = -1
                # Expand ptr: length should be n+1, fill new segment with last value for monotonicity
                old_ptr_len = int(self._arrays["answer_tok_ptr"].shape[0])
                if old_ptr_len < target_n + 1:
                    last_val = int(self._arrays["answer_tok_ptr"][old_ptr_len - 1]) if old_ptr_len > 0 else 0
                    self._arrays["answer_tok_ptr"].resize((target_n + 1,))
                    self._arrays["answer_tok_ptr"][old_ptr_len:target_n + 1] = last_val
                # Align n_samples_estimate
                self.n_samples_estimate = target_n
            
            sv = self._arrays["sample_valid"][:self.n_samples_estimate]
            if sv.any():
                first_invalid = int(np.argmax(~sv)) if not sv.all() else len(sv)
            else:
                first_invalid = 0
            if self._arrays["answer_tok_ptr"][0] == 0 and first_invalid == 0:
                self._arrays["answer_tok_ptr"][0] = 0
            
            # Expected write endpoint (consistent with pointer)
            expected_end = int(self._arrays["answer_tok_ptr"][first_invalid])
            # Current values length (may contain orphan data from previous crash)
            current_size = int(self._arrays["answer_tok_values"].shape[0])
            # If orphan data exists, truncate to ensure ptr and values sync
            if current_size > expected_end:
                self._arrays["answer_tok_values"].resize(
                    (expected_end, self.model_metadata.n_layers, self.model_metadata.hidden_dim)
                )
            self._current_ptr = expected_end
            self._resume_index = first_invalid
        else:
            self._resume_index = 0
        
        self._initialized = True
    
    def get_resume_index(self) -> int:
        """Get the resume index for continuing data collection."""
        return getattr(self, "_resume_index", 0)
    
    def save_sample(self, 
                   sample_idx: int,
                   hidden_state_data,
                   metadata: Dict[str, Any]) -> None:
        """Save a single sample's hidden state data."""
        if not self._initialized:
            raise RuntimeError("Zarr storage not initialized")
        
        # Save mean states
        if self.hidden_state_spec.need_mean and "mean_answer_hs" in self._arrays:
            self._arrays["mean_answer_hs"][sample_idx] = hidden_state_data.mean_states
        
        # Save prompt last states
        if self.hidden_state_spec.need_prompt_last and "prompt_last_hs" in self._arrays:
            self._arrays["prompt_last_hs"][sample_idx] = hidden_state_data.prompt_final_state
        
        # Save per-token states
        if (self.hidden_state_spec.need_per_token and 
            "answer_tok_values" in self._arrays and 
            hidden_state_data.per_token_states is not None and
            hidden_state_data.answer_token_count > 0):
            
            # Update pointer
            self._arrays["answer_tok_ptr"][sample_idx] = self._current_ptr
            
            # Resize and append data
            current_size = self._arrays["answer_tok_values"].shape[0]
            new_size = current_size + hidden_state_data.answer_token_count
            self._arrays["answer_tok_values"].resize((new_size, self.model_metadata.n_layers, self.model_metadata.hidden_dim))
            self._arrays["answer_tok_values"][current_size:new_size] = hidden_state_data.per_token_states
            
            self._current_ptr += hidden_state_data.answer_token_count
        else:
            # No per-token data
            self._arrays["answer_tok_ptr"][sample_idx] = self._current_ptr
        
        # Pre-write next pointer for partial reading support
        if sample_idx + 1 < self._arrays["answer_tok_ptr"].shape[0]:
            self._arrays["answer_tok_ptr"][sample_idx + 1] = self._current_ptr
        
        # Mark sample as valid
        self._arrays["sample_valid"][sample_idx] = True
        self._arrays["sample_id"][sample_idx] = metadata.get("sample_id", sample_idx)
    
    def mark_empty(self, sample_idx: int, sample_id: int) -> None:
        """Mark a sample as empty to maintain pointer alignment."""
        if "answer_tok_ptr" in self._arrays:
            self._arrays["answer_tok_ptr"][sample_idx] = self._current_ptr
            if sample_idx + 1 < self._arrays["answer_tok_ptr"].shape[0]:
                self._arrays["answer_tok_ptr"][sample_idx + 1] = self._current_ptr
        self._arrays["sample_valid"][sample_idx] = False
        self._arrays["sample_id"][sample_idx] = sample_id
    
    def finalize(self) -> None:
        """Finalize storage operations."""
        if not self._initialized:
            return
        
        # Ensure final pointer is written
        if "answer_tok_ptr" in self._arrays:
            self._arrays["answer_tok_ptr"][-1] = self._current_ptr
        
        # Consolidate metadata
        zarr.consolidate_metadata(self._store.store)
        self._initialized = False
    
    def load_mean_states(self, sample_indices: np.ndarray) -> np.ndarray:
        """Load mean states for specified samples."""
        if "mean_answer_hs" not in self._arrays:
            raise ValueError("Mean states not available")
        return self._arrays["mean_answer_hs"].oindex[sample_indices]
    
    def load_prompt_last_states(self, sample_indices: np.ndarray) -> np.ndarray:
        """Load prompt last states for specified samples."""
        if "prompt_last_hs" not in self._arrays:
            raise ValueError("Prompt last states not available")
        return self._arrays["prompt_last_hs"].oindex[sample_indices]
    
    def load_per_token_states(self, sample_indices: np.ndarray) -> list:
        """Load per-token states for specified samples."""
        if "answer_tok_values" not in self._arrays or "answer_tok_ptr" not in self._arrays:
            raise ValueError("Per-token states not available")
        
        results = []
        for idx in sample_indices:
            start = int(self._arrays["answer_tok_ptr"][idx])
            end = int(self._arrays["answer_tok_ptr"][idx + 1])
            
            if start < end:
                results.append(self._arrays["answer_tok_values"][start:end])
            else:
                results.append(np.array([]))
        
        return results
