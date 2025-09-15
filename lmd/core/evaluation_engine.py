"""Evaluation and scoring utilities."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional


class EvaluationEngine:
    """Computes evaluation metrics and scores from generation data."""
    
    @staticmethod
    def compute_generation_metrics(
        scores: List[torch.Tensor],
        generated_token_ids: Optional[List[int]] = None,
    ) -> 'GenerationMetrics':
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

        gen_tok_ids: List[int] = generated_token_ids or []
        T_ppl = min(len(scores), len(gen_tok_ids))

        for step_idx, logits in enumerate(scores):
            # logits: [1, vocab_size]
            row = logits[0].to(torch.float32) 
            probs = F.softmax(row, dim=-1)

            # 1) max prob
            max_probs.append(probs.max().item())

            # 2) entropy (bits)
            ent = -(probs * torch.log2(probs.clamp_min(1e-12))).sum().item()
            entropies.append(ent)

            # 3) chosen token log-prob (for PPL)
            if step_idx < T_ppl:
                tok_id = int(gen_tok_ids[step_idx])
                if 0 <= tok_id < probs.shape[0]:
                    p_tok = probs[tok_id].clamp_min(1e-12)
                    chosen_logprobs.append(float(torch.log(p_tok).item()))
                else:
                    chosen_logprobs.append(float(np.log(1e-12)))

        avg_max_prob = float(np.mean(max_probs)) if max_probs else 0.0
        avg_entropy = float(np.mean(entropies)) if entropies else 0.0

        if chosen_logprobs:
            ppl = float(np.exp(-float(np.mean(chosen_logprobs))))
        else:
            ppl = float('inf')

        return GenerationMetrics(
            max_probability=avg_max_prob,
            perplexity=ppl,
            entropy=avg_entropy,
        )