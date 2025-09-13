"""
LLM Trajectory Dynamics - A framework for analyzing hidden states during LLM inference.

"""

from pathlib import Path

# Project configuration
PROJECT_ROOT = Path(__file__).parent.parent
STORAGE_ROOT = PROJECT_ROOT / "storage"
CONFIG_ROOT = PROJECT_ROOT / "configs"

# Version information
__version__ = "2.0.0"
__author__ = "LLM Trajectory Dynamics Team"

# Main exports
from .core import ModelManager, DataProcessor, HiddenStateExtractor, EvaluationEngine
from .data import AnswerParser, PromptTemplateManager, DatasetRegistry
from .storage import ZarrManager, ParquetManager
from .utils.types import SampleRecord, GenerationConfig, HiddenStateSpec

__all__ = [
    # Core modules
    "ModelManager",
    "DataProcessor", 
    "HiddenStateExtractor",
    "EvaluationEngine",
    
    # Data modules
    "AnswerParser",
    "PromptTemplateManager",
    "DatasetRegistry",
    
    # Storage modules
    "ZarrManager",
    "ParquetManager",
    
    # Types
    "SampleRecord",
    "GenerationConfig",
    "HiddenStateSpec",
    
    # Configuration
    "PROJECT_ROOT",
    "STORAGE_ROOT", 
    "CONFIG_ROOT",
    "__version__",
    "__author__"
]