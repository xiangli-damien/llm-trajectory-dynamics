"""Type definitions and data structures."""

from dataclasses import dataclass
from typing import Optional


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
    n_layers: int  # Total layers including embedding
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