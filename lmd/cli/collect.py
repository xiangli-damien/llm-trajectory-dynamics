#!/usr/bin/env python3
"""Unified data collection CLI for LLM hidden states."""

import argparse
import json
import time
import torch
import gc
import signal
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from contextlib import contextmanager

from ..core import ModelManager, DataProcessor
from ..data import AnswerParser, PromptTemplateManager, DatasetRegistry
from ..storage import ZarrManager, ParquetManager
from ..utils.types import GenerationConfig, HiddenStateSpec
from ..utils.seed import set_seed


@contextmanager
def time_limit(seconds):
    """Context manager for timeout."""
    def handler(signum, frame):
        raise TimeoutError("Sample processing timeout")
    
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class DataCollector:
    """Manages data collection across models and datasets."""
    
    AVAILABLE_MODELS = [
        "Llama-3-8B-Instruct",
        "Mistral-7B-Instruct-v0.2", 
        "Qwen2-7B-Instruct"
    ]
    
    AVAILABLE_DATASETS = [
        "mmlu",
        "gsm8k",
        "mgsm", 
        "math",
        "commonsenseqa",
        "hotpotqa",
        "theoremqa",
        "belebele"
    ]
    
    def __init__(self, args):
        """Initialize data collector with arguments."""
        self.args = args
        self.data_root = Path(args.data_root)
        self.model_root = Path(args.model_root)
        self.output_root = self._setup_output_dir()
        
        # Initialize configurations
        self.gen_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False
        )
        
        self.hidden_state_spec = HiddenStateSpec(
            need_per_token=args.save_per_token,
            need_mean=args.save_mean,
            need_prompt_last=args.save_prompt_last
        )
        
        # Results tracking
        self.results = {
            "run_id": self.output_root.name,
            "start_time": datetime.now().isoformat(),
            "configuration": vars(args),
            "models": {}
        }
    
    def _setup_output_dir(self) -> Path:
        """Setup output directory with proper run ID."""
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.args.resume and self.args.run_id:
            run_dir = output_dir / self.args.run_id
            if not run_dir.exists():
                raise ValueError(f"Resume directory {run_dir} not found")
            print(f"Resuming run: {self.args.run_id}")
            return run_dir
        
        # Create new run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        models_str = f"m{len(self.args.models)}"
        datasets_str = f"d{len(self.args.datasets)}"
        run_id = f"run_{timestamp}_{models_str}_{datasets_str}"
        
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def collect_all(self):
        """Main collection loop across all models and datasets."""
        total_start = time.time()
        
        print("\n" + "="*80)
        print(f"{'DATA COLLECTION CONFIGURATION':^80}")
        print("="*80)
        print(f"Output: {self.output_root}")
        print(f"Models: {', '.join(self.args.models)}")
        print(f"Datasets: {', '.join(self.args.datasets)}")
        print(f"Max samples: {self.args.max_samples or 'All'}")
        print(f"Max tokens: {self.args.max_new_tokens}")
        print(f"Flush every: {self.args.flush_every} samples")
        print(f"Timeout: {self.args.timeout_s}s per sample")
        print(f"Output scores: {self.args.output_scores}")
        print("="*80 + "\n")
        
        for model_name in self.args.models:
            self.results["models"][model_name] = self._collect_model(model_name)
        
        # Save final results
        total_time = time.time() - total_start
        self.results["end_time"] = datetime.now().isoformat()
        self.results["total_time_seconds"] = total_time
        
        results_file = self.output_root / "collection_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self._print_summary(total_time)
    
    def _collect_model(self, model_name: str) -> Dict[str, Any]:
        """Collect data for a single model across all datasets."""
        print(f"\n{'='*60}")
        print(f"Processing Model: {model_name}")
        print(f"{'='*60}\n")
        
        # Clear GPU memory
        self._clear_gpu_memory()
        
        # Load model
        model_manager = ModelManager(
            model_name=model_name,
            model_path=str(self.model_root / model_name)
        )
        model_manager.load_model()
        
        # Configure output scores
        model_manager.set_output_scores(self.args.output_scores)
        
        print(f"✓ Model loaded: {model_manager.metadata.n_layers} layers, {model_manager.metadata.hidden_dim} dim")
        print(f"✓ Output scores: {self.args.output_scores}")
        print(f"✓ Using online metrics: {not self.args.output_scores}")
        
        # Process each dataset
        model_results = {
            "model_name": model_name,
            "start_time": datetime.now().isoformat(),
            "datasets": {}
        }
        
        for dataset_name in self.args.datasets:
            dataset_result = self._collect_dataset(model_manager, dataset_name)
            model_results["datasets"][dataset_name] = dataset_result
        
        # Cleanup
        del model_manager
        self._clear_gpu_memory()
        
        model_results["end_time"] = datetime.now().isoformat()
        return model_results
    
    def _collect_dataset(self, model_manager: ModelManager, dataset_name: str) -> Dict[str, Any]:
        """Collect data for a single dataset."""
        print(f"\nProcessing {dataset_name}...")
        
        # Setup components
        prompt_template_manager = PromptTemplateManager()
        answer_parser = AnswerParser(dataset_name)
        dataset_registry = DatasetRegistry(self.data_root)
        
        # Load samples
        samples = dataset_registry.load_samples(
            dataset_name=dataset_name,
            language=self.args.language,
            max_samples=self.args.max_samples
        )
        n_samples = len(samples)
        
        # Initialize processor
        data_processor = DataProcessor(
            model_manager=model_manager,
            prompt_template_manager=prompt_template_manager,
            answer_parser=answer_parser,
            hidden_state_spec=self.hidden_state_spec
        )
        
        # Setup storage
        zarr_path = self.output_root / "zarr" / f"{model_manager.model_name}__{dataset_name}__{self.args.language}"
        zarr_manager = ZarrManager(
            storage_path=zarr_path,
            model_metadata=model_manager.metadata,
            hidden_state_spec=self.hidden_state_spec,
            n_samples_estimate=n_samples,
            chunk_t=self.args.zarr_chunk_t
        )
        
        parquet_path = self.output_root / "parquet" / model_manager.model_name / dataset_name
        parquet_manager = ParquetManager(parquet_path)
        
        # Check resume position
        start_idx = 0
        if self.args.resume:
            start_idx = zarr_manager.get_resume_index()
            if self.args.verbose and start_idx > 0:
                print(f"Resuming {dataset_name} at sample index {start_idx}")
        
        # Override with start_index if specified
        if self.args.start_index is not None:
            start_idx = max(start_idx, self.args.start_index)
        
        # Parse skip indices
        skip_set = set()
        if self.args.skip_indices.strip():
            skip_set = {int(x) for x in self.args.skip_indices.split(",") if x.strip().isdigit()}
        
        # Process samples
        processed_samples = []
        correct_count = 0
        error_count = 0
        processed_count = 0  # Track successfully processed samples
        
        with tqdm(total=n_samples, desc=f"{dataset_name:15}", unit="sample", initial=start_idx) as pbar:
            for i, sample in enumerate(samples):
                if i < start_idx:
                    continue
                
                if i in skip_set:
                    zarr_manager.mark_empty(sample_idx=i, sample_id=sample.sample_id)
                    pbar.update(1)
                    continue
                    
                try:
                    # Process sample with timeout
                    with time_limit(self.args.timeout_s):
                        processed = data_processor.process_sample(sample, self.gen_config)
                    
                    processed_samples.append(processed)
                    processed_count += 1  # Increment for successful processing
                    
                    if processed.is_correct:
                        correct_count += 1
                    
                    # Save to Zarr
                    zarr_manager.save_sample(
                        sample_idx=i,
                        hidden_state_data=processed.hidden_state_data,
                        metadata={
                            "sample_id": processed.sample_id,
                            "is_correct": processed.is_correct,
                            "answer_token_count": processed.hidden_state_data.answer_token_count
                        }
                    )
                    
                    # Free large arrays immediately after saving
                    processed.hidden_state_data.per_token_states = None
                    processed.hidden_state_data.mean_states = None
                    processed.hidden_state_data.prompt_final_state = None
                    
                    # Update progress with correct accuracy calculation
                    acc = correct_count / processed_count * 100 if processed_count > 0 else 0
                    pbar.set_postfix({'Acc': f"{acc:.1f}%", 'Err': error_count})
                    
                    # Periodic flush to bound memory
                    if len(processed_samples) >= self.args.flush_every:
                        parquet_manager.save_metadata(processed_samples)
                        parquet_manager.save_outputs(processed_samples)
                        processed_samples.clear()
                    
                except TimeoutError:
                    error_count += 1
                    zarr_manager.mark_empty(sample_idx=i, sample_id=sample.sample_id)
                    if self.args.verbose:
                        print(f"\nTimeout on sample {i}")
                        
                except Exception as e:
                    error_count += 1
                    zarr_manager.mark_empty(sample_idx=i, sample_id=sample.sample_id)
                    if self.args.verbose:
                        print(f"\nError on sample {i}: {e}")
                
                pbar.update(1)
                
                # Aggressive memory cleanup
                if (i + 1) % 25 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        # Final flush
        if processed_samples:
            parquet_manager.save_metadata(processed_samples)
            parquet_manager.save_outputs(processed_samples)
        
        # Finalize storage
        zarr_manager.finalize()
        
        # Return statistics with correct accuracy
        accuracy = correct_count / processed_count if processed_count > 0 else 0
        print(f"✓ {dataset_name}: {correct_count}/{processed_count} correct ({accuracy:.1%}), {error_count} errors")
        
        return {
            "n_samples": n_samples,
            "n_processed": processed_count,
            "n_correct": correct_count,
            "n_errors": error_count,
            "accuracy": accuracy
        }
    
    def _clear_gpu_memory(self):
        """Clear GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _print_summary(self, total_time: float):
        """Print collection summary."""
        print("\n" + "="*80)
        print(f"{'COLLECTION COMPLETE':^80}")
        print("="*80)
        
        total_samples = 0
        total_correct = 0
        total_processed = 0
        
        for model_name, model_data in self.results["models"].items():
            print(f"\n{model_name}:")
            for dataset_name, stats in model_data.get("datasets", {}).items():
                n_processed = stats.get("n_processed", 0)
                n_correct = stats.get("n_correct", 0)
                acc = stats.get("accuracy", 0)
                print(f"  {dataset_name:15}: {n_correct:4}/{n_processed:4} ({acc:.1%})")
                total_samples += stats.get("n_samples", 0)
                total_processed += n_processed
                total_correct += n_correct
        
        overall_acc = total_correct / total_processed * 100 if total_processed > 0 else 0
        print(f"\nTotal: {total_correct}/{total_processed} ({overall_acc:.1%})")
        print(f"Time: {total_time/60:.1f} minutes")
        print(f"Output: {self.output_root}")
        print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect hidden states from LLM inference")
    
    # Paths
    parser.add_argument("--data-root", type=str, default="storage/datasets")
    parser.add_argument("--model-root", type=str, default="storage/models")
    parser.add_argument("--output-dir", type=str, default="storage/runs")
    
    # Selection
    parser.add_argument("--models", nargs="+", default=DataCollector.AVAILABLE_MODELS,
                       choices=DataCollector.AVAILABLE_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DataCollector.AVAILABLE_DATASETS,
                       choices=DataCollector.AVAILABLE_DATASETS)
    
    # Parameters
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--language", type=str, default="en")
    
    # Hidden states
    parser.add_argument("--save-per-token", dest="save_per_token", action="store_true", default=True)
    parser.add_argument("--no-save-per-token", dest="save_per_token", action="store_false")
    parser.add_argument("--save-mean", dest="save_mean", action="store_true", default=True)
    parser.add_argument("--no-save-mean", dest="save_mean", action="store_false")
    parser.add_argument("--save-prompt-last", dest="save_prompt_last", action="store_true", default=True)
    parser.add_argument("--no-save-prompt-last", dest="save_prompt_last", action="store_false")
    
    # Memory management
    parser.add_argument("--flush-every", type=int, default=100,
                       help="Flush parquet every N samples to bound memory")
    parser.add_argument("--timeout-s", type=int, default=1800,
                       help="Timeout per sample in seconds")
    parser.add_argument("--zarr-chunk-t", type=int, default=64,
                       help="Chunk size for token axis in zarr")
    parser.add_argument("--output-scores", action="store_true", default=False,
                       help="Output scores for metrics (uses lots of memory)")
    
    # Resume and skip
    parser.add_argument("--start-index", type=int, default=None,
                       help="Override resume and start from this index")
    parser.add_argument("--skip-indices", type=str, default="",
                       help="Comma-separated indices to skip")
    
    # Options
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id", type=str, help="Resume specific run")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Run collection
    collector = DataCollector(args)
    collector.collect_all()


if __name__ == "__main__":
    main()