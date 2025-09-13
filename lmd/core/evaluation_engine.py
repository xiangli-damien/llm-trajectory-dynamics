"""Evaluation and scoring utilities."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any


class EvaluationEngine:
    """Computes evaluation metrics and scores from generation data."""
    
    @staticmethod
    def compute_generation_metrics(scores: List[torch.Tensor]) -> 'GenerationMetrics':
        """Compute generation metrics from per-step scores."""
        from .data_processor import GenerationMetrics
        
        if not scores:
            return GenerationMetrics(
                max_probability=0.0,
                perplexity=np.inf,
                entropy=0.0
            )
        
        max_probs = []
        entropies = []
        
        for logits in scores:
            # Convert logits to probabilities
            probs = F.softmax(logits[0], dim=-1)
            
            # Maximum probability
            max_prob = probs.max().item()
            max_probs.append(max_prob)
            
            # Entropy
            entropy = -(probs * torch.log2(probs + 1e-12)).sum().item()
            entropies.append(entropy)
        
        # Aggregate metrics
        avg_max_prob = np.mean(max_probs)
        perplexity = np.exp(-np.mean(np.log(np.maximum(max_probs, 1e-12))))
        avg_entropy = np.mean(entropies)
        
        return GenerationMetrics(
            max_probability=avg_max_prob,
            perplexity=perplexity,
            entropy=avg_entropy
        )