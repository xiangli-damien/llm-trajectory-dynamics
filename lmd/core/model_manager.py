"""Model management and generation utilities."""

import torch
import inspect
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.generation.logits_process import LogitsProcessorList
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
    scores: Optional[List[torch.Tensor]]
    finish_reason: str


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
        self._output_scores = False
        
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

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self._config,
            torch_dtype=self.dtype,
            device_map=self.device_map,
            trust_remote_code=True
        )
        self._model.eval()

        gen_cfg = self._model.generation_config
        try:
            gen_cfg.update(
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=self._output_scores,
                top_k=None,
            )
        except AttributeError:
            for k, v in dict(
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=self._output_scores,
                top_k=None,
            ).items():
                if hasattr(gen_cfg, k):
                    setattr(gen_cfg, k, v)
        
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
        # Probe actual hidden state layer count
        self._metadata.n_layers = self._probe_hidden_state_layers()
    
    def set_output_scores(self, output_scores: bool) -> None:
        """Set whether to output scores during generation."""
        self._output_scores = output_scores
        if self._model and hasattr(self._model, 'generation_config'):
            try:
                self._model.generation_config.output_scores = output_scores
            except:
                pass
    
    def _extract_metadata(self) -> ModelMetadata:
        """Extract model metadata."""
        config = self._config
        
        # Get base layer count
        n_layers = getattr(config, "num_hidden_layers", None) or \
                  getattr(config, "n_layer", None) or \
                  getattr(config, "n_layers", None)
        
        # Placeholder - will be updated by probe
        n_layers_with_emb = n_layers + 1 if n_layers else 0
        
        return ModelMetadata(
            model_name=self.model_name,
            n_layers=n_layers_with_emb,
            hidden_dim=getattr(config, "hidden_size", None) or \
                      getattr(config, "n_embd", None) or \
                      getattr(config, "d_model", None),
            vocab_size=getattr(config, "vocab_size", None),
            model_type=getattr(config, "model_type", None)
        )
    
    def _probe_hidden_state_layers(self) -> int:
        """Probe actual hidden state layer count from model."""
        try:
            # Create minimal input
            tok = self._tokenizer("probe", return_tensors="pt").to(self._model.device)
            
            # Run forward pass to get hidden states
            with torch.inference_mode():
                out = self._model(**tok, output_hidden_states=True, use_cache=False, return_dict=True)
            
            # Count actual layers returned
            hs = out.hidden_states
            if isinstance(hs, (list, tuple)):
                return len(hs)
        except Exception:
            pass
        
        # Fallback to config + 1
        cfg_layers = getattr(self._config, "num_hidden_layers", None) or \
                    getattr(self._config, "n_layer", None) or \
                    getattr(self._config, "n_layers", None) or 0
        return cfg_layers + 1
    
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
    def metadata(self) -> ModelMetadata:
        """Get the model metadata."""
        if self._metadata is None:
            raise RuntimeError("Metadata not available. Call load_model() first.")
        return self._metadata
    
    def generate_sequence(self, input_ids: torch.Tensor, gen_config: GenerationConfig,
                         logits_processors: Optional[List[Any]] = None) -> GenerationResult:
        """Generate text sequence with hidden states and optional scores.
        
        Args:
            input_ids: Input token ids
            gen_config: Generation configuration
            logits_processors: Optional list of logits processors
            
        Returns:
            GenerationResult with generated tokens and hidden states
        """
        device = next(self.model.parameters()).device
        
        # Create attention mask for safety
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        
        # Get termination tokens
        terminators = self._get_termination_tokens()
        
        # Generate with hidden states and optionally scores
        with torch.inference_mode():
            sig = inspect.signature(self.model.generate)

            base_kwargs = dict(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=gen_config.max_new_tokens,
                temperature=gen_config.temperature if gen_config.do_sample else None,
                top_p=gen_config.top_p if gen_config.do_sample else None,
                do_sample=gen_config.do_sample,
                eos_token_id=terminators,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=self._output_scores,
                output_hidden_states=True,
            )
            
            # Add logits processors if provided
            if logits_processors:
                base_kwargs["logits_processor"] = LogitsProcessorList(logits_processors)

            if "generation_config" in sig.parameters:
                base_kwargs["generation_config"] = self.model.generation_config

            if (not gen_config.do_sample) and "top_k" in sig.parameters:
                base_kwargs["top_k"] = None

            gen_output = self.model.generate(**base_kwargs)
        
        # Extract results
        sequences = gen_output.sequences
        input_length = input_ids.shape[1]
        generated_length = sequences.shape[1] - input_length
        
        generated_tokens = sequences[:, input_length:]
        hidden_states = gen_output.hidden_states
        if generated_length > 0 and (hidden_states is None or len(hidden_states) == 0):
            raise RuntimeError("Hidden states not returned by generate(). Check generation config.")

        scores = gen_output.scores if self._output_scores else None
        
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
            finish_reason=finish_reason
        )
    
    def _get_termination_tokens(self) -> List[int]:
        """Get list of termination tokens."""
        terminators = [self.tokenizer.eos_token_id]
        
        # Add model-specific terminators
        special_terminators = ["<|eot_id|>", "<|im_end|>", "</s>"]
        for term in special_terminators:
            try:
                term_id = self.tokenizer.convert_tokens_to_ids(term)
                if term_id is not None and term_id not in terminators:
                    terminators.append(term_id)
            except:
                pass
        
        return terminators
    
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