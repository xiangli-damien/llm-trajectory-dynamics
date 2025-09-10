"""
Core modules for LLM Trajectory Dynamics.

This package contains the core business logic for collecting and analyzing
hidden states from language models during inference.
"""

from .model_manager import ModelManager
from .data_processor import DataProcessor
from .hidden_state_extractor import HiddenStateExtractor
from .evaluation_engine import EvaluationEngine

__all__ = [
    "ModelManager",
    "DataProcessor", 
    "HiddenStateExtractor",
    "EvaluationEngine"
]
