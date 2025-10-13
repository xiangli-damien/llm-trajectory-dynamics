from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

class MetricDirection(Enum):
    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"

class DataMode(Enum):
    STATE_MEAN = "mean"
    STATE_PROMPT_LAST = "prompt_last"
    TOKEN_STREAM = "token"

@dataclass
class RunConfig:
    run_dir: Path
    model: str
    dataset: str
    language: str = "en"
    
    def __post_init__(self):
        self.run_dir = Path(self.run_dir)

@dataclass
class FilterSpec:
    exclude_extraction_failures: bool = False
    exclude_truncated: bool = False
    min_answer_tokens: int = 0
    max_answer_tokens: Optional[int] = None
    require_valid_in_zarr: bool = True
    verbose: bool = False

@dataclass
class LayerSpec:
    drop_embedding: bool = False
    exclude_last_n: int = 0
    stride: int = 1
    use_layers: Optional[List[int]] = None
    layer_range: Optional[Tuple[int, int]] = None

@dataclass
class MetricConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SubsetSpec:
    mode: str = "all"
    size: Optional[int] = None
    frac: Optional[float] = None
    seed: int = 42
    stratify_by: Optional[str] = None

@dataclass
class ExperimentConfig:
    data_mode: DataMode
    layer_spec: LayerSpec
    metrics: List[MetricConfig]
    subset_spec: Optional[SubsetSpec] = None
    token_agg: Optional[str] = "mean"
    token_agg_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RunInfo:
    n_samples: int
    n_correct: int
    n_incorrect: int
    accuracy: float
    n_layers: int
    hidden_dim: int
    model: str
    dataset: str
    language: str
    n_filtered: int

@dataclass
class RunData:
    arrays: Dict[str, Any]
    metadata: pd.DataFrame
    outputs: pd.DataFrame
    correct_indices: np.ndarray
    incorrect_indices: np.ndarray
    info: RunInfo
    labels: np.ndarray
    row_indices: np.ndarray

@dataclass
class MetricOutput:
    name: str
    scores: Dict[str, np.ndarray]
    directions: Dict[str, MetricDirection]
    cache_state: Optional[Dict[str, Any]] = None