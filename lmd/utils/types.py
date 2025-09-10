"""Type definitions and data structures for LLM Trajectory Dynamics."""

from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple, Union
import numpy as np
import torch


@dataclass
class SampleRecord:
    """Record representing a single data sample."""
    sample_id: int
    question: str
    answer_gt: str
    dataset: str
    language: str = "en"
    answer_type: Optional[str] = None


@dataclass
class ModelMetadata:
    """Metadata about a loaded model."""
    model_name: str
    n_layers: int
    hidden_dim: int
    vocab_size: int
    model_type: str


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


@dataclass
class HiddenStateSpec:
    """Specification for hidden state extraction."""
    need_per_token: bool = True
    need_mean: bool = True
    need_prompt_last: bool = True


@dataclass
class CollectionConfig:
    """Configuration for data collection run."""
    model_name: str
    dataset_name: str
    language: str
    run_id: str
    generation_config: GenerationConfig
    hidden_state_spec: HiddenStateSpec
    max_samples: Optional[int] = None
    seed: int = 42


@dataclass
class StorageConfig:
    """Configuration for data storage."""
    zarr_dtype: str = "float32"
    compression_level: int = 5
    chunk_size: int = 1024
    enable_coe_export: bool = False


# Protocol definitions for extensibility
from typing import Protocol


class ModelManagerProtocol(Protocol):
    """Protocol for model management."""
    
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        ...
    
    def generate_sequence(self, input_ids: torch.Tensor, config: GenerationConfig) -> Any:
        """Generate text sequence with hidden states."""
        ...


class StorageManagerProtocol(Protocol):
    """Protocol for storage management."""
    
    def save_sample(self, sample_data: Any) -> None:
        """Save a single sample."""
        ...
    
    def finalize(self) -> None:
        """Finalize storage operations."""
        ...


class AnswerParserProtocol(Protocol):
    """Protocol for answer parsing."""
    
    def parse_and_label(self, generated_text: str, sample: SampleRecord) -> Dict[str, Any]:
        """Parse generated text and check correctness."""
        ...
