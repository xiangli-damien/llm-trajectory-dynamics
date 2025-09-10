"""Main CLI for data collection."""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from tqdm import tqdm

# Optional psutil import for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from ..core import ModelManager, DataProcessor, HiddenStateExtractor, EvaluationEngine
from ..data import AnswerParser, PromptTemplateManager, DatasetRegistry
from ..storage import ZarrManager, ParquetManager
from ..utils.types import GenerationConfig, HiddenStateSpec, CollectionConfig, StorageConfig
from ..utils.seed import set_seed


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Collect hidden states from LLM inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument("--model-name", type=str, required=True, help="Model identifier")
    parser.add_argument("--model-dir", type=str, required=True, help="Model directory path")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="Model data type")
    parser.add_argument("--device-map", type=str, default="auto", help="Device mapping strategy")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--data-dir", type=str, default="storage/datasets", help="Data directory")
    parser.add_argument("--language", type=str, default="en", help="Language code")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to process")
    
    # Generation configuration
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling")
    
    # Hidden state configuration
    parser.add_argument("--save-per-token", dest="save_per_token", action="store_true")
    parser.add_argument("--no-save-per-token", dest="save_per_token", action="store_false")
    parser.set_defaults(save_per_token=True)
    
    parser.add_argument("--save-mean", dest="save_mean", action="store_true")
    parser.add_argument("--no-save-mean", dest="save_mean", action="store_false")
    parser.set_defaults(save_mean=True)
    
    parser.add_argument("--save-prompt-last", dest="save_prompt_last", action="store_true")
    parser.add_argument("--no-save-prompt-last", dest="save_prompt_last", action="store_false")
    parser.set_defaults(save_prompt_last=True)
    
    # Storage configuration
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--run-id", type=str, help="Run identifier (auto-generated if not provided)")
    parser.add_argument("--zarr-dtype", type=str, default="float32", choices=["float32", "float16"], help="Zarr storage data type")
    
    # Other options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Generate run ID if not provided
    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create output directory
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize components
        print("Initializing components...")
        
        # Model manager
        model_manager = ModelManager(
            model_name=args.model_name,
            model_path=args.model_dir,
            device_map=args.device_map,
            dtype=args.dtype
        )
        model_manager.load_model()
        
        # Data components
        prompt_template_manager = PromptTemplateManager()
        answer_parser = AnswerParser(args.dataset)
        dataset_registry = DatasetRegistry(Path(args.data_dir))
        
        # Load samples
        print(f"Loading samples from {args.dataset}...")
        samples = dataset_registry.load_samples(
            dataset_name=args.dataset,
            language=args.language,
            max_samples=args.max_samples
        )
        print(f"Loaded {len(samples)} samples")
        
        # Configuration
        gen_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample
        )
        
        hidden_state_spec = HiddenStateSpec(
            need_per_token=args.save_per_token,
            need_mean=args.save_mean,
            need_prompt_last=args.save_prompt_last
        )
        
        # Data processor
        data_processor = DataProcessor(
            model_manager=model_manager,
            prompt_template_manager=prompt_template_manager,
            answer_parser=answer_parser,
            hidden_state_spec=hidden_state_spec
        )
        
        # Storage managers
        zarr_path = output_dir / "zarr" / f"{args.model_name}__{args.dataset}__{args.language}"
        zarr_manager = ZarrManager(
            storage_path=zarr_path,
            model_metadata=model_manager.metadata,
            hidden_state_spec=hidden_state_spec,
            n_samples_estimate=len(samples),
            per_token_dtype="float16" if args.zarr_dtype == "float16" else "float32",
            mean_dtype="float32",
            prompt_dtype="float32"
        )
        
        parquet_manager = ParquetManager(output_dir)
        
        # Create JSONL metadata file
        meta_dir = output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_file = (meta_dir / f"{args.model_name}_{args.dataset}_samples.jsonl").open("a", encoding="utf-8")
        
        # Process samples with progress tracking
        print("Processing samples...")
        processed_samples = []
        error_count = 0
        start_time = time.time()
        
        # Initialize progress bar
        progress_bar = tqdm(
            total=len(samples),
            desc="Processing",
            unit="sample",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        
        for i, sample in enumerate(samples):
            try:
                processed = data_processor.process_sample(sample, gen_config)
                processed_samples.append(processed)
                
                # Save to storage
                zarr_manager.save_sample(
                    sample_idx=i,
                    hidden_state_data=processed.hidden_state_data,
                    metadata={
                        "sample_id": processed.sample_id,
                        "dataset": processed.dataset,
                        "language": processed.language,
                        "question": processed.question,
                        "ground_truth": processed.answer_gt,
                        "generated_text": processed.generated_text,
                        "extracted_answer": processed.extracted_answer,
                        "is_correct": processed.is_correct,
                        "finish_reason": processed.finish_reason,
                        "input_length": processed.input_length,
                        "answer_token_count": processed.hidden_state_data.answer_token_count
                    }
                )
                
                # Write to JSONL metadata
                meta_row = {
                    "sample_id": processed.sample_id,
                    "dataset": processed.dataset,
                    "language": processed.language,
                    "model": processed.model,
                    "question": processed.question,
                    "answer_gt": processed.answer_gt,
                    "prompt": processed.prompt,
                    "generated_text": processed.generated_text,
                    "generated_text_raw": processed.generated_text_raw,
                    "extracted_answer": processed.extracted_answer,
                    "is_correct": processed.is_correct,
                    "finish_reason": processed.finish_reason,
                    "metrics": {
                        "max_probability": processed.generation_metrics.max_probability,
                        "perplexity": processed.generation_metrics.perplexity,
                        "entropy": processed.generation_metrics.entropy
                    },
                    "token_counts": {
                        "prompt_length": processed.input_length,
                        "generated_length": len(processed.answer_token_ids),
                        "answer_token_count": processed.hidden_state_data.answer_token_count
                    },
                    "hidden_states": {
                        "has_per_token": processed.hidden_state_data.per_token_states is not None,
                        "answer_token_count": processed.hidden_state_data.answer_token_count
                    }
                }
                meta_file.write(json.dumps(meta_row, ensure_ascii=False) + "\n")
                
                # Update progress metrics
                success_rate = len(processed_samples) / (i + 1) * 100
                memory_usage = f"{psutil.Process().memory_info().rss / 1024 / 1024:.0f}MB" if PSUTIL_AVAILABLE else "N/A"
                
                progress_bar.set_postfix({
                    'Sample': sample.sample_id,
                    'Success': f"{success_rate:.1f}%",
                    'Errors': error_count,
                    'Memory': memory_usage
                })
                
            except Exception as e:
                error_count += 1
                # Mark empty sample to maintain pointer alignment
                zarr_manager.mark_empty(sample_idx=i, sample_id=sample.sample_id)
                if args.verbose:
                    print(f"\nError processing sample {sample.sample_id}: {e}")
            
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Finalize storage
        print("Finalizing storage...")
        zarr_manager.finalize()
        
        # Save metadata and outputs
        parquet_manager.save_metadata(processed_samples)
        parquet_manager.save_outputs(processed_samples)
        
        # Close JSONL metadata file
        meta_file.close()
        
        # Save run configuration
        config = {
            "run_id": run_id,
            "model_name": args.model_name,
            "dataset": args.dataset,
            "language": args.language,
            "n_samples": len(processed_samples),
            "generation_config": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "do_sample": args.do_sample
            },
            "hidden_state_spec": {
                "need_per_token": args.save_per_token,
                "need_mean": args.save_mean,
                "need_prompt_last": args.save_prompt_last
            },
            "timestamp": datetime.now().isoformat(),
            "seed": args.seed
        }
        
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Print summary
        total_time = time.time() - start_time
        correct_count = sum(1 for s in processed_samples if s.is_correct)
        accuracy = correct_count / len(processed_samples) if processed_samples else 0.0
        avg_tokens = sum(s.hidden_state_data.answer_token_count for s in processed_samples) / len(processed_samples) if processed_samples else 0.0
        
        print(f"\nCollection Complete!")
        print(f"Results saved to: {output_dir}")
        print(f"Total samples: {len(samples)}")
        print(f"Processed: {len(processed_samples)}")
        print(f"Errors: {error_count}")
        print(f"Success rate: {len(processed_samples)/len(samples)*100:.1f}%")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Avg tokens: {avg_tokens:.1f}")
        print(f"Time: {total_time/60:.1f}min ({len(samples)/total_time:.1f} samples/sec)")
        
    except Exception as e:
        print(f"Error during collection: {e}")
        raise


if __name__ == "__main__":
    main()
