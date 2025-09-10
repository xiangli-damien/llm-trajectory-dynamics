"""Evaluation and scoring utilities."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any
from scipy.stats import entropy as H
from dataclasses import dataclass


@dataclass
class GenerationMetrics:
    """Metrics computed from generation scores."""
    max_probability: float
    perplexity: float
    entropy: float


class EvaluationEngine:
    """Computes evaluation metrics and scores from generation data."""
    
    @staticmethod
    def compute_generation_metrics(scores: List[torch.Tensor]) -> GenerationMetrics:
        """Compute generation metrics from per-step scores."""
        if not scores:
            return GenerationMetrics(
                max_probability=np.nan,
                perplexity=np.nan,
                entropy=np.nan
            )
        
        max_probs = []
        entropies = []
        
        for logits in scores:
            # Convert logits to probabilities
            probs = F.softmax(logits[0], dim=-1)  # [V]
            
            # Maximum probability for this step
            max_probs.append(float(probs.max().item()))
            
            # Entropy for this step (base 2)
            entropies.append(float(H(probs.detach().cpu().numpy(), base=2)))
        
        # Compute aggregate metrics
        avg_max_prob = float(np.mean(max_probs))
        perplexity = float(-np.mean(np.log(np.maximum(max_probs, 1e-12))))
        avg_entropy = float(np.mean(entropies))
        
        return GenerationMetrics(
            max_probability=avg_max_prob,
            perplexity=perplexity,
            entropy=avg_entropy
        )
    
    @staticmethod
    def compute_accuracy_metrics(predictions: List[bool]) -> Dict[str, float]:
        """Compute accuracy metrics from prediction results."""
        if not predictions:
            return {"accuracy": 0.0, "total_samples": 0}
        
        correct_count = sum(predictions)
        total_count = len(predictions)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct_samples": correct_count,
            "total_samples": total_count
        }
    
    @staticmethod
    def compute_generation_statistics(metrics_list: List[GenerationMetrics]) -> Dict[str, Any]:
        """Compute statistics across multiple generation metrics."""
        if not metrics_list:
            return {}
        
        max_probs = [m.max_probability for m in metrics_list if not np.isnan(m.max_probability)]
        perplexities = [m.perplexity for m in metrics_list if not np.isnan(m.perplexity)]
        entropies = [m.entropy for m in metrics_list if not np.isnan(m.entropy)]
        
        stats = {}
        
        if max_probs:
            stats["max_probability"] = {
                "mean": float(np.mean(max_probs)),
                "std": float(np.std(max_probs)),
                "min": float(np.min(max_probs)),
                "max": float(np.max(max_probs))
            }
        
        if perplexities:
            stats["perplexity"] = {
                "mean": float(np.mean(perplexities)),
                "std": float(np.std(perplexities)),
                "min": float(np.min(perplexities)),
                "max": float(np.max(perplexities))
            }
        
        if entropies:
            stats["entropy"] = {
                "mean": float(np.mean(entropies)),
                "std": float(np.std(entropies)),
                "min": float(np.min(entropies)),
                "max": float(np.max(entropies))
            }
        
        return stats
