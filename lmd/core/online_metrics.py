"""Online metrics collection without storing logits."""

import math
import torch
from transformers.generation.logits_process import LogitsProcessor


class OnlineMetricsProcessor(LogitsProcessor):
    """Collect max probability, entropy and perplexity online without storage."""
    
    def __init__(self, greedy: bool = True):
        """Initialize online metrics processor.
        
        Args:
            greedy: Whether generation is greedy (do_sample=False)
        """
        self.greedy = greedy
        self.n_steps = 0
        self.sum_max_prob = 0.0
        self.sum_entropy = 0.0
        self.sum_logprob = 0.0
    
    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits for current generation step.
        
        Args:
            input_ids: Input token ids [batch_size, sequence_length]
            scores: Logits for current step [batch_size, vocab_size]
            
        Returns:
            Unchanged scores tensor
        """
        # Only process first batch element
        logits = scores[0].to(torch.float32)
        probs = torch.softmax(logits, dim=-1)
        
        # Compute max probability
        max_prob = torch.max(probs).item()
        self.sum_max_prob += max_prob
        
        # Compute entropy in bits
        entropy = -(probs * torch.log2(probs.clamp_min(1e-12))).sum().item()
        self.sum_entropy += entropy
        
        # For greedy generation, track chosen token's log probability
        if self.greedy:
            chosen_token = torch.argmax(logits).item()
            chosen_prob = probs[chosen_token].clamp_min(1e-12)
            self.sum_logprob += torch.log(chosen_prob).item()
        
        self.n_steps += 1
        
        # Return unchanged scores
        return scores
    
    def finalize(self) -> dict:
        """Compute final averaged metrics.
        
        Returns:
            Dictionary with max_probability, entropy, and perplexity
        """
        if self.n_steps == 0:
            return {
                'max_probability': 0.0,
                'entropy': 0.0,
                'perplexity': float('inf')
            }
        
        avg_max_prob = self.sum_max_prob / self.n_steps
        avg_entropy = self.sum_entropy / self.n_steps
        
        # Compute perplexity from average negative log likelihood
        if self.greedy and self.n_steps > 0:
            avg_nll = -self.sum_logprob / self.n_steps
            perplexity = math.exp(avg_nll)
        else:
            perplexity = float('inf')
        
        return {
            'max_probability': avg_max_prob,
            'entropy': avg_entropy,
            'perplexity': perplexity
        }