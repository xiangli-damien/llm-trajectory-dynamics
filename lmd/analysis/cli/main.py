import argparse
import json
import sys
import os
from pathlib import Path
import pandas as pd
import yaml
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from ..core.types import (
    RunConfig, FilterSpec, LayerSpec, ExperimentConfig, 
    MetricConfig, SubsetSpec, DataMode
)
from ..core.context import RunContext
from ..core.registry import MetricRegistry
from ..core.subset import select_subset, apply_subset_to_data
from ..data.run_io import load_run_data
from ..models.lm_head import load_lm_head
from ..experiments.runner import ExperimentRunner
from ..metrics import ndr, rownull, robust_late, coe, fisher_exact
from ..metrics import cids, gmm_discriminative
from ..metrics import subspace_learned, fusion
from ..metrics import pca_classifier, supcon_tail

def setup_registry() -> MetricRegistry:
    registry = MetricRegistry()
    registry.register("ndr", ndr.NDR)
    registry.register("rownull", rownull.RowNull)
    registry.register("robust_late", robust_late.RobustLatePhase)
    registry.register("coe", coe.CoE)
    registry.register("fisher_exact", fisher_exact.FisherExact)
    registry.register("cids", cids.CIDS)
    registry.register("gmm_disc", gmm_discriminative.GMMDiscriminative)
    registry.register("subspace_fda", subspace_learned.TailFDA)
    registry.register("spca_sup", subspace_learned.SPCASupervised)
    registry.register("fusion_rankavg", fusion.FusionRankAvg)
    registry.register("pca_cls", pca_classifier.PCABasisClassifier)
    registry.register("supcon_tail", supcon_tail.SupConTail)
    return registry

def load_config(config_path: Optional[Path]) -> ExperimentConfig:
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        return parse_config_dict(data)
    return None

def parse_config_dict(data: dict) -> ExperimentConfig:
    layer_spec = LayerSpec(**data.get('layer_spec', {}))
    
    metrics = []
    for m in data.get('metrics', []):
        if isinstance(m, str):
            metrics.append(MetricConfig(name=m))
        else:
            metrics.append(MetricConfig(**m))
    
    data_mode_str = data.get('data_mode', 'state_mean')
    data_mode_map = {
        'state_mean': DataMode.STATE_MEAN,
        'mean': DataMode.STATE_MEAN,
        'state_prompt_last': DataMode.STATE_PROMPT_LAST,
        'prompt_last': DataMode.STATE_PROMPT_LAST,
        'token': DataMode.TOKEN_STREAM,
        'token_stream': DataMode.TOKEN_STREAM
    }
    data_mode = data_mode_map.get(data_mode_str, DataMode.STATE_MEAN)
    
    subset_spec = None
    if 'subset' in data:
        subset_spec = SubsetSpec(**data['subset'])
    
    return ExperimentConfig(
        data_mode=data_mode,
        layer_spec=layer_spec,
        metrics=metrics,
        subset_spec=subset_spec,
        token_agg=data.get('token_agg', 'median'),
        token_agg_params=data.get('token_agg_params', {})
    )

def main():
    parser = argparse.ArgumentParser(description="LLM Trajectory Analysis")
    
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--model-path", type=Path, default=None)
    
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    parser.add_argument("--data-mode", choices=["state_mean", "state_prompt_last", "token"],
                       default="state_mean")
    parser.add_argument("--metrics", nargs="+", 
                       default=[
                           "cids", "gmm_disc",
                           "subspace_fda", "spca_sup",
                           "pca_cls",
                           "fusion_rankavg",
                           "ndr", "rownull", "robust_late", "coe"
                       ])
    
    parser.add_argument("--drop-embedding", action="store_true")
    parser.add_argument("--exclude-last-n", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--use-layers", nargs="+", type=int, default=None)
    parser.add_argument("--layer-range", nargs=2, type=int, default=None)
    
    parser.add_argument("--subset-mode", choices=["all", "correct", "incorrect", "random", "balanced"],
                       default="all")
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--subset-frac", type=float, default=None)
    parser.add_argument("--subset-seed", type=int, default=42)
    
    parser.add_argument("--token-agg", default="median")
    parser.add_argument("--token-agg-q", type=float, default=0.75)
    parser.add_argument("--token-agg-k", type=int, default=8)
    
    parser.add_argument("--save-json", type=Path)
    parser.add_argument("--save-parquet", type=Path)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if args.config:
        config = load_config(args.config)
    else:
        layer_spec = LayerSpec(
            drop_embedding=args.drop_embedding,
            exclude_last_n=args.exclude_last_n,
            stride=args.stride,
            use_layers=args.use_layers,
            layer_range=tuple(args.layer_range) if args.layer_range else None
        )
        
        metrics = [MetricConfig(name=m) for m in args.metrics]
        
        data_mode_map = {
            'state_mean': DataMode.STATE_MEAN,
            'state_prompt_last': DataMode.STATE_PROMPT_LAST,
            'token': DataMode.TOKEN_STREAM
        }
        
        subset_spec = SubsetSpec(
            mode=args.subset_mode,
            size=args.subset_size,
            frac=args.subset_frac,
            seed=args.subset_seed
        ) if args.subset_mode != "all" else None
        
        token_agg_params = {}
        if args.token_agg == "quantile":
            token_agg_params['q'] = args.token_agg_q
        elif args.token_agg == "topk_mean":
            token_agg_params['k'] = args.token_agg_k
        
        config = ExperimentConfig(
            data_mode=data_mode_map[args.data_mode],
            layer_spec=layer_spec,
            metrics=metrics,
            subset_spec=subset_spec,
            token_agg=args.token_agg,
            token_agg_params=token_agg_params
        )
    
    run_cfg = RunConfig(args.run_dir, args.model, args.dataset, args.language)
    filter_spec = FilterSpec(verbose=args.verbose)
    data = load_run_data(run_cfg, filter_spec)
    
    if config.subset_spec is not None and config.subset_spec.mode != "all":
        sel_idx = select_subset(data.labels, data.metadata, config.subset_spec)
        data = apply_subset_to_data(data, sel_idx)
        if args.verbose:
            print(f"Subset: {config.subset_spec.mode}, size={len(data.labels)}")
    
    lm_head = None
    if args.model_path:
        if args.verbose:
            print(f"Loading LM head from {args.model_path}")
        lm_head = load_lm_head(args.model_path, var_ratio=0.95, exact_mode=False)
    
    ctx = RunContext(config=run_cfg, data=data, lm_head=lm_head)
    
    registry = setup_registry()
    runner = ExperimentRunner(registry)
    
    results = runner.run(ctx, config, n_jobs=args.n_jobs, verbose=args.verbose)
    
    if args.save_json:
        save_results_json(results, args.save_json, args.verbose)
    
    if args.save_parquet:
        save_results_parquet(results, args.save_parquet, args.verbose)

def save_results_json(results: dict, path: Path, verbose: bool):
    output = {
        "meta": results["meta"],
        "evals": results["evals"],
        "scores": {k: v.tolist() if hasattr(v, 'tolist') else v 
                  for k, v in results["scores"].items() if v.ndim == 1}
    }
    
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    
    if verbose:
        print(f"Results saved to {path}")

def save_results_parquet(results: dict, path: Path, verbose: bool):
    scores_1d = {k: v for k, v in results["scores"].items() if v.ndim == 1}
    if scores_1d:
        df = pd.DataFrame(scores_1d)
        df.to_parquet(path, index=False)
        if verbose:
            print(f"Scores saved to {path}")

if __name__ == "__main__":
    main()