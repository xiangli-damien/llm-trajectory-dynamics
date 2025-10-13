"""Data processing pipeline for LLM trajectory analysis."""

import torch
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from .model_manager import ModelManager
from .hidden_state_extractor import HiddenStateExtractor, HiddenStateData
from .evaluation_engine import EvaluationEngine
from .online_metrics import OnlineMetricsProcessor
from ..data.answer_parser import AnswerParser
from ..data.prompt_templates import PromptTemplateManager
from ..utils.types import SampleRecord, GenerationConfig, HiddenStateSpec


@dataclass
class GenerationMetrics:
    """Metrics computed from generation scores."""
    max_probability: float
    perplexity: float
    entropy: float


@dataclass
class ProcessedSample:
    """Result of processing a single sample."""
    sample_id: int
    prompt: str
    generated_text: str
    extracted_answer: str
    is_correct: bool
    hidden_state_data: HiddenStateData
    generation_metrics: GenerationMetrics
    finish_reason: str
    answer_token_ids: List[int]
    input_length: int
    # Metadata fields
    dataset: str
    language: str
    question: str
    answer_gt: str
    model: str
    answer_type: Optional[str] = None


class DataProcessor:
    """Main data processing pipeline for trajectory analysis."""
    
    def __init__(self, 
                 model_manager: ModelManager,
                 prompt_template_manager: PromptTemplateManager,
                 answer_parser: AnswerParser,
                 hidden_state_spec: HiddenStateSpec):
        """Initialize data processor."""
        self.model_manager = model_manager
        self.prompt_template_manager = prompt_template_manager
        self.answer_parser = answer_parser
        # Pass model metadata to hidden state extractor
        self.hidden_state_extractor = HiddenStateExtractor(hidden_state_spec, model_manager.metadata)
        self.evaluation_engine = EvaluationEngine()
    
    def process_sample(self, sample: SampleRecord, gen_config: GenerationConfig) -> ProcessedSample:
        """Process a single sample through the complete pipeline."""
        # Generate prompt
        prompt = self.prompt_template_manager.get_prompt(
            dataset=sample.dataset,
            question=sample.question,
            answer_type=sample.answer_type,
            language=sample.language
        )
        
        # Apply chat template
        input_ids = self._apply_chat_template(prompt)
        
        # Check if we should use online metrics collection
        use_online_metrics = not getattr(self.model_manager, "_output_scores", False)
        metrics_processor = None
        logits_processors = None
        
        if use_online_metrics:
            # Use online metrics processor for greedy generation
            metrics_processor = OnlineMetricsProcessor(greedy=(not gen_config.do_sample))
            logits_processors = [metrics_processor]
        
        # Generate sequence with hidden states
        generation_result = self.model_manager.generate_sequence(
            input_ids, gen_config, logits_processors=logits_processors
        )
        
        # Decode generated text
        generated_text = self.model_manager.tokenizer.decode(
            generation_result.generated_tokens[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Extract prompt-only hidden states for prompt_final_state
        prompt_hidden_states = None
        if self.hidden_state_extractor.spec.need_prompt_last:
            prompt_hidden_states = self._get_prompt_hidden_states(input_ids)
        
        # Extract hidden states
        hidden_state_data = self.hidden_state_extractor.extract_from_generation(
            generation_result, prompt_hidden_states
        )
        
        # Compute generation metrics
        if use_online_metrics and metrics_processor is not None:
            # Get metrics from online processor
            metrics_dict = metrics_processor.finalize()
            generation_metrics = GenerationMetrics(
                max_probability=metrics_dict['max_probability'],
                perplexity=metrics_dict['perplexity'],
                entropy=metrics_dict['entropy']
            )
        else:
            # Use traditional evaluation engine
            generation_metrics = self.evaluation_engine.compute_generation_metrics(
                generation_result.scores,
                generation_result.generated_tokens[0].tolist()
            )
        
        # Parse answer and check correctness
        parse_result = self.answer_parser.parse_and_label(generated_text, sample)
        
        return ProcessedSample(
            sample_id=sample.sample_id,
            prompt=prompt,
            generated_text=generated_text,
            extracted_answer=parse_result["extracted_answer"],
            is_correct=parse_result["correct"],
            hidden_state_data=hidden_state_data,
            generation_metrics=generation_metrics,
            finish_reason=generation_result.finish_reason,
            answer_token_ids=generation_result.generated_tokens[0].tolist(),
            input_length=generation_result.input_length,
            # Metadata
            dataset=sample.dataset,
            language=sample.language,
            question=sample.question,
            answer_gt=sample.answer_gt,
            model=self.model_manager.model_name,
            answer_type=sample.answer_type
        )
    
    def _apply_chat_template(self, prompt: str) -> torch.Tensor:
        """Apply chat template to prompt."""
        messages = [{"role": "user", "content": prompt}]
        return self.model_manager.apply_chat_template(messages)
    
    def _get_prompt_hidden_states(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Get hidden states from prompt-only forward pass."""
        device = next(self.model_manager.model.parameters()).device
        
        with torch.inference_mode():
            outputs = self.model_manager.model(
                input_ids=input_ids.to(device),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )
            # Return all layers including embedding
            return [h.to(torch.float32).cpu() for h in outputs.hidden_states]