#!/usr/bin/env python3
"""Complete data collection script with model and dataset selection."""

import argparse
import json
import time
import torch
import sys
import signal
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from lmd.core import ModelManager, DataProcessor
from lmd.data import AnswerParser, PromptTemplateManager, DatasetRegistry
from lmd.storage import ZarrManager, ParquetManager
from lmd.utils.types import GenerationConfig, HiddenStateSpec, SampleRecord
from lmd.utils.seed import set_seed

# Available models and datasets
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
    "theoremqa"
]

def print_separator(title: str, char: str = "=", width: int = 80):
    """Print a formatted separator."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

def run_one_model(
    model_name: str,
    datasets: List[str],
    data_root: Path,
    model_root: Path,
    out_root: Path,
    max_samples_per_dataset: Optional[int],
    gen_cfg: GenerationConfig,
    hs_spec: HiddenStateSpec,
    language: str = "en",
    verbose: bool = False,
    resume: bool = False
) -> Dict[str, Any]:
    """Run data collection for one model across all datasets."""
    print_separator(f"Processing Model: {model_name}")
    
    # Clear GPU cache and memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU Memory cleared before loading {model_name}")
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Load model
    print(f"Loading model: {model_name}")
    model_manager = ModelManager(
        model_name=model_name, 
        model_path=str(model_root / model_name)
    )
    model_manager.load_model()
    
    # Initialize processors
    prompt_template_manager = PromptTemplateManager()
    data_processor = DataProcessor(
        model_manager=model_manager,
        prompt_template_manager=prompt_template_manager,
        answer_parser=AnswerParser("gsm8k"),  # Placeholder, will be updated per dataset
        hidden_state_spec=hs_spec,
        per_token_dtype="float16",
        stat_dtype="float32"
    )
    
    # Initialize JSONL metadata writer
    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Results for this model
    model_summary = {
        "model_name": model_name,
        "start_time": datetime.now().isoformat(),
        "datasets": {}
    }
    
    total_samples = 0
    total_processed = 0
    total_errors = 0
    total_correct = 0
    
    # Process each dataset
    for dataset_name in datasets:
        print_separator(f"Dataset: {dataset_name}", char="-", width=60)
        
        try:
            # Load samples using DatasetRegistry
            dataset_registry = DatasetRegistry(data_root)
            samples = dataset_registry.load_samples(
                dataset_name=dataset_name,
                language=language,
                max_samples=max_samples_per_dataset
            )
            n_samples = len(samples)
            print(f"Loaded {n_samples} samples from {dataset_name}")
            
            # Update answer parser for this dataset
            data_processor.answer_parser = AnswerParser(dataset_name)
            
            # Create Zarr storage for this dataset
            zarr_path = out_root / "zarr" / f"{model_name}__{dataset_name}__{language}"
            zarr_manager = ZarrManager(
                storage_path=zarr_path,
                model_metadata=model_manager.metadata,
                hidden_state_spec=hs_spec,
                n_samples_estimate=n_samples,
                per_token_dtype="float16",
                mean_dtype="float32",
                prompt_dtype="float32"
            )
            
            start_idx = zarr_manager.get_resume_index()
            if start_idx > 0:
                print(f"Resuming {dataset_name} from sample index {start_idx}/{n_samples}")
            
            # Create Parquet storage for this dataset
            parquet_manager = ParquetManager(out_root / "parquet" / model_name / dataset_name)
            
            # Create JSONL metadata file for this dataset
            dataset_meta_dir = meta_dir / model_name / dataset_name
            dataset_meta_dir.mkdir(parents=True, exist_ok=True)
            meta_file = (dataset_meta_dir / "samples.jsonl").open("a", encoding="utf-8")
            
            # Process samples with progress tracking
            processed_samples = []
            dataset_processed = 0
            dataset_errors = 0
            dataset_correct = 0
            
            # Create progress bar with real-time output
            progress_bar = tqdm(
                total=n_samples,
                desc=f"{model_name} - {dataset_name}",
                unit="sample",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                file=sys.stderr,  # Use stderr for real-time display
                dynamic_ncols=True,
                miniters=1,  # Update on every iteration
                initial=start_idx
            )
            
            for i, sample in enumerate(samples):
                if i < start_idx:
                    continue
                try:
                    # Process sample
                    processed = data_processor.process_sample(sample, gen_cfg)
                    processed_samples.append(processed)
                    
                    # Save to Zarr
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
                    
                    dataset_processed += 1
                    dataset_correct += 1 if processed.is_correct else 0
                    
                    # Update progress
                    success_rate = dataset_processed / (i + 1) * 100
                    accuracy = dataset_correct / dataset_processed * 100 if dataset_processed > 0 else 0
                    
                    progress_bar.set_postfix({
                        'Success': f"{success_rate:.1f}%",
                        'Accuracy': f"{accuracy:.1f}%",
                        'Errors': dataset_errors
                    })
                    
                    # Force flush for real-time display
                    sys.stderr.flush()
                    
                    # Periodic memory cleanup every 10 samples
                    if (i + 1) % 10 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                except Exception as e:
                    dataset_errors += 1
                    # Mark empty sample to maintain pointer alignment
                    zarr_manager.mark_empty(sample_idx=i, sample_id=getattr(sample, "sample_id", i))
                    if verbose:
                        print(f"\nError processing sample {i}: {e}")
                
                progress_bar.update(1)
            
            progress_bar.close()
            
            # Finalize Zarr storage
            zarr_manager.finalize()
            
            # Save Parquet files
            parquet_manager.save_metadata(processed_samples)
            parquet_manager.save_outputs(processed_samples)
            
            # Close JSONL metadata file
            meta_file.close()
            
            # Calculate final metrics
            accuracy = dataset_correct / dataset_processed if dataset_processed > 0 else 0
            success_rate = dataset_processed / n_samples * 100 if n_samples > 0 else 0
            
            print(f"{dataset_name} completed: {dataset_processed}/{n_samples} samples, {accuracy:.1%} accuracy")
            
            # Update totals
            total_samples += n_samples
            total_processed += dataset_processed
            total_errors += dataset_errors
            total_correct += dataset_correct
            
            # Store dataset results
            model_summary["datasets"][dataset_name] = {
                "dataset_name": dataset_name,
                "total_samples": n_samples,
                "processed_samples": dataset_processed,
                "error_count": dataset_errors,
                "correct_count": dataset_correct,
                "accuracy": accuracy,
                "success_rate": success_rate
            }
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            model_summary["datasets"][dataset_name] = {
                "dataset_name": dataset_name,
                "error": str(e),
                "total_samples": 0,
                "processed_samples": 0,
                "error_count": 0,
                "correct_count": 0,
                "accuracy": 0.0,
                "success_rate": 0.0
            }
    
    # Model completion summary
    model_summary["end_time"] = datetime.now().isoformat()
    model_summary["total_samples"] = total_samples
    model_summary["total_processed"] = total_processed
    model_summary["total_errors"] = total_errors
    model_summary["total_correct"] = total_correct
    model_summary["overall_accuracy"] = total_correct / total_processed if total_processed > 0 else 0
    model_summary["overall_success_rate"] = total_processed / total_samples * 100 if total_samples > 0 else 0
    
    print(f"\n{model_name} completed: {total_processed}/{total_samples} samples, {model_summary['overall_accuracy']:.1%} accuracy")
    
    # Clean up model and memory
    del model_manager
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete
        print(f"GPU Memory cleared after {model_name}")
    
    # Force garbage collection
    import gc
    gc.collect()
    
    return model_summary

def main():
    """Main collection function."""
    global zarr_manager_global
    
    def _graceful_exit(signum, frame):
        """Graceful exit handler"""
        try:
            if 'zarr_manager_global' in globals() and zarr_manager_global is not None:
                zarr_manager_global.finalize()
                print(f"\nGraceful exit: Finalized Zarr storage")
        except Exception as e:
            print(f"\nError during graceful exit: {e}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)
    
    parser = argparse.ArgumentParser(description="Collect hidden states for selected models and datasets")
    
    # Required arguments
    parser.add_argument("--data_root", type=str, default="storage/datasets", help="Path to datasets directory")
    parser.add_argument("--model_root", type=str, default="storage/models", help="Path to models directory")
    parser.add_argument("--output_dir", type=str, default="storage/runs", help="Output directory")
    
    # Model and dataset selection
    parser.add_argument("--models", nargs="+", choices=AVAILABLE_MODELS, default=AVAILABLE_MODELS, help="Models to process")
    parser.add_argument("--datasets", nargs="+", choices=AVAILABLE_DATASETS, default=AVAILABLE_DATASETS, help="Datasets to process")
    
    # Processing parameters
    parser.add_argument("--max_samples_per_dataset", type=int, default=None, help="Maximum samples per dataset (None for full collection)")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum new tokens to generate")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling (1.0 for greedy)")
    
    # Full collection mode
    parser.add_argument("--full_collection", action="store_true", help="Enable full collection mode (process all samples)")
    
    # Storage options
    parser.add_argument("--language", type=str, default="en", help="Language code")
    
    # Resume options
    parser.add_argument("--run_id", type=str, help="Existing run id to resume")
    parser.add_argument("--resume", action="store_true", help="Resume into existing run dir")
    
    # Other options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Handle full collection mode
    if args.full_collection:
        args.max_samples_per_dataset = None
        print("Full collection mode enabled - processing all samples")
    
    # Convert paths
    data_root = Path(args.data_root)
    model_root = Path(args.model_root)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle resume mode
    if args.resume and args.run_id:
        out_root = output_dir / args.run_id
        if not out_root.exists():
            print(f"Error: Resume directory {out_root} not found")
            sys.exit(1)
        run_id = args.run_id
        print(f"Resuming existing run: {run_id}")
    else:
        # Generate run ID
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        mode_suffix = "full" if args.full_collection else f"m{len(args.models)}-d{len(args.datasets)}"
        run_id = f"run_{timestamp}__{mode_suffix}__seed{args.seed}"
        out_root = output_dir / run_id
        out_root.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print_separator("Data Collection Configuration")
    print(f"Run ID: {run_id}")
    print(f"Data root: {data_root}")
    print(f"Model root: {model_root}")
    print(f"Output dir: {out_root}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Max samples per dataset: {args.max_samples_per_dataset or 'All'}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Top-p: {args.top_p}")
    print(f"Storage: per-token=float16, mean/prompt=float32")
    print(f"Language: {args.language}")
    print(f"Seed: {args.seed}")
    print(f"Full collection: {args.full_collection}")
    
    # Force flush for immediate display
    sys.stdout.flush()
    
    # Initialize configurations
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=args.top_p,
        do_sample=False
    )
    
    hidden_state_spec = HiddenStateSpec(
        need_per_token=True,
        need_mean=True,
        need_prompt_last=True
    )
    
    # Start collection
    start_time = time.time()
    all_results = {
        "run_id": run_id,
        "start_time": datetime.now().isoformat(),
        "configuration": vars(args),
        "models": {}
    }
    
    # Calculate total tasks for global progress
    total_tasks = len(args.models) * len(args.datasets)
    completed_tasks = 0
    
    # Create status file for monitoring
    status_file = out_root / "collection_status.json"
    
    def update_status(status_data):
        """Update status file for monitoring"""
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
    
    print_separator("Starting Data Collection")
    print(f"Total tasks: {total_tasks} (models: {len(args.models)}, datasets: {len(args.datasets)})")
    print(f"Status file: {status_file}")
    
    # Initial status
    update_status({
        "status": "starting",
        "total_tasks": total_tasks,
        "completed_tasks": 0,
        "current_model": None,
        "current_dataset": None,
        "start_time": datetime.now().isoformat()
    })
    
    # Process each model
    for model_name in args.models:
        try:
            # Update status
            update_status({
                "status": "running",
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "current_model": model_name,
                "current_dataset": None,
                "start_time": datetime.now().isoformat()
            })
            
            model_results = run_one_model(
                model_name=model_name,
                datasets=args.datasets,
                data_root=data_root,
                model_root=model_root,
                out_root=out_root,
                max_samples_per_dataset=args.max_samples_per_dataset,
                gen_cfg=gen_config,
                hs_spec=hidden_state_spec,
                language=args.language,
                verbose=args.verbose,
                resume=args.resume
            )
            
            all_results["models"][model_name] = model_results
            completed_tasks += len(args.datasets)
            print(f"Model {model_name} completed ({completed_tasks}/{total_tasks} tasks)")
            
            # Update status
            update_status({
                "status": "running",
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "current_model": None,
                "current_dataset": None,
                "start_time": datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            all_results["models"][model_name] = {
                "model_name": model_name,
                "error": str(e),
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat()
            }
            completed_tasks += len(args.datasets)
    
    # Final summary
    end_time = time.time()
    total_time = end_time - start_time
    
    # Update final status
    update_status({
        "status": "completed",
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "current_model": None,
        "current_dataset": None,
        "start_time": datetime.now().isoformat(),
        "end_time": datetime.now().isoformat(),
        "total_time_seconds": total_time
    })
    
    all_results["end_time"] = datetime.now().isoformat()
    all_results["total_time_seconds"] = total_time
    
    # Calculate overall statistics
    total_samples = sum(m.get("total_samples", 0) for m in all_results["models"].values())
    total_processed = sum(m.get("total_processed", 0) for m in all_results["models"].values())
    total_errors = sum(m.get("total_errors", 0) for m in all_results["models"].values())
    total_correct = sum(m.get("total_correct", 0) for m in all_results["models"].values())
    
    print_separator("Collection Complete")
    print(f"Run ID: {run_id}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total samples: {total_samples}")
    print(f"Processed: {total_processed}")
    print(f"Errors: {total_errors}")
    print(f"Correct: {total_correct}")
    print(f"Success rate: {total_processed/total_samples*100:.1f}%" if total_samples > 0 else "N/A")
    print(f"Overall accuracy: {total_correct/total_processed*100:.1f}%" if total_processed > 0 else "N/A")
    print(f"Results saved to: {out_root}")
    
    # Save final results
    results_file = out_root / "collection_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Final results saved to: {results_file}")

if __name__ == "__main__":
    main()