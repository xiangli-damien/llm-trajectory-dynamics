"""Model management and generation utilities."""

import torch
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from dataclasses import dataclass

from ..utils.types import GenerationConfig, ModelMetadata


@dataclass
class GenerationResult:
    """Result of text generation with hidden states and scores."""
    input_ids: torch.Tensor
    generated_tokens: torch.Tensor
    full_sequence: torch.Tensor
    input_length: int
    generated_length: int
    hidden_states: List[Tuple[torch.Tensor, ...]]
    scores: List[torch.Tensor]
    finish_reason: str
    terminators: List[int]


class ModelManager:
    """Manages language model loading, configuration, and generation."""
    
    def __init__(self, model_name: str, model_path: str, device_map: str = "auto", dtype: str = "auto"):
        """Initialize model manager."""
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.device_map = device_map
        self.dtype = self._resolve_dtype(dtype)
        
        self._model = None
        self._tokenizer = None
        self._config = None
        self._metadata = None
        
    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        """Resolve torch dtype from string."""
        if dtype == "auto":
            if torch.cuda.is_available():
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.float32
        return getattr(torch, dtype)
    
    def load_model(self) -> None:
        """Load the model, tokenizer, and configuration."""
        # Load configuration
        self._config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self._config.output_hidden_states = True
        
        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self._config,
            torch_dtype=self.dtype,
            device_map=self.device_map,
            trust_remote_code=True
        )
        self._model.eval()
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        # Set padding token
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        
        # Extract metadata
        self._metadata = self._extract_metadata()
    
    def _extract_metadata(self) -> ModelMetadata:
        """Extract model metadata."""
        def _get_attr(cfg, *candidates):
            for key in candidates:
                val = getattr(cfg, key, None)
                if val is not None:
                    return val
            return None
        
        return ModelMetadata(
            model_name=self.model_name,
            n_layers=_get_attr(self._config, "num_hidden_layers", "n_layer", "n_layers"),
            hidden_dim=_get_attr(self._config, "hidden_size", "n_embd", "d_model"),
            vocab_size=_get_attr(self._config, "vocab_size"),
            model_type=_get_attr(self._config, "model_type")
        )
    
    @property
    def model(self) -> AutoModelForCausalLM:
        """Get the loaded model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model
    
    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get the loaded tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")
        return self._tokenizer
    
    @property
    def config(self) -> AutoConfig:
        """Get the model configuration."""
        if self._config is None:
            raise RuntimeError("Config not loaded. Call load_model() first.")
        return self._config
    
    @property
    def metadata(self) -> ModelMetadata:
        """Get the model metadata."""
        if self._metadata is None:
            raise RuntimeError("Metadata not available. Call load_model() first.")
        return self._metadata
    
    def generate_sequence(self, input_ids: torch.Tensor, gen_config: GenerationConfig) -> GenerationResult:
        """Generate text sequence with hidden states and scores."""
        device = next(self.model.parameters()).device
        
        # Create attention mask to avoid pad_token=eos_token issues
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        # Collect termination tokens
        terminators = self._get_termination_tokens()
        
        # Generate with hidden states and scores
        with torch.inference_mode():
            gen_output = self.model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=gen_config.max_new_tokens,
                temperature=gen_config.temperature if gen_config.do_sample else None,
                top_p=gen_config.top_p if gen_config.do_sample else None,
                do_sample=gen_config.do_sample,
                eos_token_id=terminators,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True
            )
        
        # Extract results
        sequences = gen_output.sequences
        input_length = input_ids.shape[1]
        generated_length = sequences.shape[1] - input_length
        
        generated_tokens = sequences[:, input_length:]
        hidden_states = gen_output.hidden_states
        scores = gen_output.scores
        
        # Determine finish reason
        finish_reason = self._determine_finish_reason(
            generated_tokens, generated_length, gen_config.max_new_tokens, terminators
        )
        
        return GenerationResult(
            input_ids=input_ids,
            generated_tokens=generated_tokens,
            full_sequence=sequences,
            input_length=input_length,
            generated_length=generated_length,
            hidden_states=hidden_states,
            scores=scores,
            finish_reason=finish_reason,
            terminators=terminators
        )
    
    def _get_termination_tokens(self) -> List[int]:
        """Get list of termination tokens."""
        terminators = [self.tokenizer.eos_token_id]
        
        if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
            additional_terminators = ["<|eot_id|>", "<|im_end|>", "</s>"]
            for term in additional_terminators:
                try:
                    term_id = self.tokenizer.convert_tokens_to_ids(term)
                    if term_id is not None:
                        terminators.append(term_id)
                except Exception:
                    pass
        
        # Remove duplicates and None values
        return [t for i, t in enumerate(terminators) if t is not None and t not in terminators[:i]]
    
    def _determine_finish_reason(self, generated_tokens: torch.Tensor, 
                               generated_length: int, max_tokens: int, 
                               terminators: List[int]) -> str:
        """Determine why generation finished."""
        if generated_length == 0:
            return "empty"
        
        last_token = generated_tokens[0, -1].item()
        if last_token in terminators:
            return "eos"
        elif generated_length >= max_tokens:
            return "length"
        else:
            return "other"
    
    def apply_chat_template(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        """Apply chat template to messages."""
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
        else:
            # Fallback for models without chat template
            text = " ".join([msg["content"] for msg in messages])
            return self.tokenizer(text, return_tensors="pt").input_ids
