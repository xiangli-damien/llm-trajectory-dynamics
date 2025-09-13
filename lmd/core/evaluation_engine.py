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
                perplexity=float('inf'),
                entropy=0.0,
            )

        max_probs: List[float] = []
        entropies: List[float] = []
        chosen_logprobs: List[float] = []

        # Effective steps for PPL (need both scores and generated tokens)
        T_ppl = min(len(scores), len(generated_token_ids)) if generated_token_ids else 0

        for step_idx, logits in enumerate(scores):
            # logits: [1, vocab_size] (batch_size==1). Convert to float32 for stability.
            row = logits[0].to(torch.float32)
            probs = F.softmax(row, dim=-1)

            # 1) Maximum probability at this step
            max_probs.append(probs.max().item())

            # 2) Entropy (bits) at this step
            ent = -(probs * torch.log2(probs.clamp_min(1e-12))).sum().item()
            entropies.append(ent)

            # 3) Chosen token log-prob for PPL
            if step_idx < T_ppl:
                tok_id = int(generated_token_ids[step_idx])
                if 0 <= tok_id < probs.shape[0]:
                    p_tok = probs[tok_id].clamp_min(1e-12)
                    chosen_logprobs.append(float(torch.log(p_tok).item()))
                else:
                    # Token id out of range (shouldn't happen); treat as tiny prob
                    chosen_logprobs.append(float(np.log(1e-12)))

        avg_max_prob = float(np.mean(max_probs)) if max_probs else 0.0
        avg_entropy = float(np.mean(entropies)) if entropies else 0.0

        if chosen_logprobs:
            # mean negative log-prob over generated tokens
            mean_nll = -float(np.mean(chosen_logprobs))  # natural log
            ppl = float(np.exp(mean_nll))
        else:
            ppl = float('inf')

        return GenerationMetrics(
            max_probability=avg_max_prob,
            perplexity=ppl,
            entropy=avg_entropy,
        )