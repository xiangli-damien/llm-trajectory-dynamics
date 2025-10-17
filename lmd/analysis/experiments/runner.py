import numpy as np
from typing import Dict, List, Any, Optional
from joblib import Parallel, delayed
from ..core.context import RunContext
from ..core.loader import StateLoader, TokenStreamer
from ..core.layers import compute_shared_cache
from ..core.aggregators import Aggregators
from ..core.evaluate import evaluate_metric
from ..core.types import ExperimentConfig, MetricOutput, DataMode, MetricConfig, MetricDirection
from ..core.registry import MetricRegistry

class ExperimentRunner:
    def __init__(self, registry: MetricRegistry):
        self.registry = registry
    
    def run(self, ctx: RunContext, config: ExperimentConfig, 
            n_jobs: int = -1, verbose: bool = False) -> Dict[str, Any]:
        
        if verbose:
            print(f"Running experiment: {config.data_mode.value}")
        
        results = {
            "scores": {},
            "evals": {},
            "meta": self._build_meta(ctx, config)
        }
        
        if config.data_mode == DataMode.TOKEN_STREAM:
            return self._run_token_mode(ctx, config, results, n_jobs, verbose)
        else:
            return self._run_state_mode(ctx, config, results, n_jobs, verbose)
    
    def _run_state_mode(self, ctx: RunContext, config: ExperimentConfig, 
                       results: Dict, n_jobs: int, verbose: bool) -> Dict[str, Any]:
        
        mode = "mean" if config.data_mode == DataMode.STATE_MEAN else "prompt_last"
        loader = StateLoader(ctx.arrays, ctx.sample_rows, config.layer_spec)
        states = loader.load_states(mode)
        
        if verbose:
            print(f"Loaded states: shape={states.shape}")
            print("Building shared cache...")
        
        cache = compute_shared_cache(states, ctx.lm_head, var_ratio=None)
        ctx.shared_cache.update(cache)
        
        results['layerwise'] = self._layerwise_significance(ctx, states)
        
        metric_order = self._get_execution_order(config.metrics)
        
        for metric_configs in metric_order:
            if verbose:
                print(f"Computing {len(metric_configs)} metrics...")
            
            valid_configs = []
            for cfg in metric_configs:
                instance = self.registry.create_instance(cfg)
                if instance.requires_lm_head and ctx.lm_head is None:
                    if verbose:
                        print(f"Skipping {cfg.name}: requires lm_head but none provided")
                    continue
                valid_configs.append(cfg)
            
            if not valid_configs:
                continue
            
            if n_jobs == 1:
                outputs = [self._compute_metric(ctx, states, cfg) for cfg in valid_configs]
            else:
                outputs = Parallel(n_jobs=n_jobs, backend="threading")(
                    delayed(self._compute_metric)(ctx, states, cfg) for cfg in valid_configs
                )
            
            for output in outputs:
                if output:
                    self._merge_output(ctx, results, output)
        
        if verbose:
            self._print_summary(results)
        
        return results
    
    def _run_token_mode(self, ctx: RunContext, config: ExperimentConfig, 
                   results: Dict, n_jobs: int, verbose: bool) -> Dict[str, Any]:
    
        agg_fn = Aggregators.get(config.token_agg, **config.token_agg_params)
        
        last_k = config.token_agg_params.get('last_k', None)
        streamer = TokenStreamer(ctx.arrays, ctx.sample_rows, 
                                config.layer_spec, batch_size=128, last_k=last_k)
        
        metric_order = self._get_execution_order(config.metrics)
        
        all_scores = {}
        all_directions = {}
        
        for metric_configs in metric_order:
            if verbose:
                print(f"Processing {len(metric_configs)} metrics in token mode...")
            
            metric_instances = []
            for cfg in metric_configs:
                instance = self.registry.create_instance(cfg)
                if instance.requires_lm_head and ctx.lm_head is None:
                    if verbose:
                        print(f"Skipping {cfg.name}: requires lm_head but none provided")
                    continue
                if "token" not in instance.supported_modes:
                    if verbose:
                        print(f"Skipping {cfg.name}: does not support token mode")
                    continue
                metric_instances.append(instance)
            
            if not metric_instances:
                continue
            
            token_results = {m.name: {} for m in metric_instances}
            
            write_position = 0
            for batch_indices, sequences in streamer.stream():
                start_idx = write_position
                end_idx = start_idx + len(batch_indices)
                
                for instance in metric_instances:
                    try:
                        output = instance.compute_token(ctx, sequences, agg_fn)
                        ctx.set_metric_state(instance.name, output.cache_state)
                        for key, values in output.scores.items():
                            if key not in token_results[instance.name]:
                                token_results[instance.name][key] = np.zeros(len(ctx.labels), dtype=np.float32)
                            token_results[instance.name][key][start_idx:end_idx] = values
                        all_directions.update(output.directions)
                    except NotImplementedError:
                        if verbose:
                            print(f"Warning: {instance.name} claims token support but not implemented")
                        continue
                
                write_position = end_idx
            
            for name, scores in token_results.items():
                all_scores.update(scores)
        
        results["scores"] = all_scores
        for key, scores in all_scores.items():
            if scores.ndim == 1 and key in all_directions:
                results["evals"][key] = evaluate_metric(ctx.labels, scores, all_directions[key])
        
        if verbose:
            self._print_summary(results)
        
        return results
    
    def _layerwise_significance(self, ctx: RunContext, states: np.ndarray) -> Dict:
        diag = {}
        labels = ctx.labels.astype(int)
        cache = ctx.shared_cache
        
        def auc_effect(per_step_values, direction=MetricDirection.HIGHER_BETTER):
            K = per_step_values.shape[1]
            out = []
            for k in range(K):
                ev = evaluate_metric(labels, per_step_values[:, k], direction)
                auc = ev['auroc']
                eff = auc if auc >= 0.5 else (1.0 - auc)
                out.append({'layer': k, 'auroc': auc, 'effect': eff})
            return out
        
        if 'dr' in cache and cache['dr'].ndim == 2:
            diag['dr'] = auc_effect(cache['dr'], MetricDirection.HIGHER_BETTER)
        
        if 'dq' in cache and 'dh_energy' in cache:
            dq = cache['dq']
            if dq.ndim == 3:
                row_e = np.sum(dq * dq, axis=2)
                tot_e = cache['dh_energy']
                ratio = row_e / (tot_e + 1e-12)
                diag['row_ratio'] = auc_effect(ratio, MetricDirection.HIGHER_BETTER)
                
                N, L_diff, _ = dq.shape
                best_auroc = -1.0
                best_k = L_diff
                for test_k in range(2, min(L_diff + 1, 21)):
                    tail_ratio = np.mean(ratio[:, -test_k:], axis=1)
                    ev = evaluate_metric(labels, tail_ratio, MetricDirection.HIGHER_BETTER)
                    if ev['auroc'] > best_auroc:
                        best_auroc = ev['auroc']
                        best_k = test_k
                diag['row_ratio_best_k'] = best_k
                diag['row_ratio_best_auroc'] = best_auroc
        
        if 'dq' in cache and cache['dq'].ndim == 3:
            dq_norm = np.linalg.norm(cache['dq'], axis=2)
            diag['dq_norm'] = auc_effect(dq_norm, MetricDirection.HIGHER_BETTER)
        
        if 'dlognorms' in cache and cache['dlognorms'].ndim == 2:
            diag['dlognorms'] = auc_effect(cache['dlognorms'], MetricDirection.LOWER_BETTER)
        
        return diag
    
    def _compute_metric(self, ctx: RunContext, states: np.ndarray, 
                       config: MetricConfig) -> Optional[MetricOutput]:
        try:
            instance = self.registry.create_instance(config)
            if instance.requires_lm_head and ctx.lm_head is None:
                return None
            return instance.compute_state(ctx, states)
        except Exception as e:
            print(f"Error computing metric {config.name}: {e}")
            return None
    
    def _merge_output(self, ctx: RunContext, results: Dict, output: MetricOutput):
        for key, scores in output.scores.items():
            results["scores"][key] = scores
            if scores.ndim == 1:
                direction = output.directions.get(key)
                if direction:
                    results["evals"][key] = evaluate_metric(ctx.labels, scores, direction)
        
        if output.cache_state is not None:
            ctx.set_metric_state(output.name, output.cache_state)
    
    def _get_execution_order(self, configs: List[MetricConfig]) -> List[List[MetricConfig]]:
        basic_metrics = []
        combo_metrics = []
        
        for cfg in configs:
            info = self.registry.get_metric_info(cfg.name)
            if info.get('dependencies'):
                combo_metrics.append(cfg)
            else:
                basic_metrics.append(cfg)
        
        return [basic_metrics, combo_metrics] if basic_metrics else [combo_metrics]
    
    def _build_meta(self, ctx: RunContext, config: ExperimentConfig) -> Dict[str, Any]:
        return {
            "model": ctx.config.model,
            "dataset": ctx.config.dataset,
            "language": ctx.config.language,
            "n_samples": len(ctx.labels),
            "n_correct": int(ctx.correct_mask.sum()),
            "n_incorrect": int(ctx.incorrect_mask.sum()),
            "accuracy": float(ctx.correct_mask.mean()) if len(ctx.labels) > 0 else 0.0,
            "data_mode": config.data_mode.value,
            "layer_spec": vars(config.layer_spec)
        }
    
    def _print_summary(self, results: Dict):
        print(f"Computed {len(results['scores'])} metrics")
        if results['evals']:
            best_metric = max(results['evals'].items(), key=lambda x: x[1]['auroc'])
            print(f"Best metric: {best_metric[0]} (AUROC={best_metric[1]['auroc']:.4f})")
        
        if 'layerwise' in results and results['layerwise']:
            print("\nLayerwise significance (top effects):")
            for key, layer_results in results['layerwise'].items():
                if key == 'row_ratio_best_k':
                    print(f"  Optimal tail window for row_ratio: {layer_results} layers")
                elif key == 'row_ratio_best_auroc':
                    print(f"  Best row_ratio AUROC: {layer_results:.4f}")
                elif isinstance(layer_results, list) and layer_results:
                    sorted_layers = sorted(layer_results, key=lambda x: x['effect'], reverse=True)[:3]
                    print(f"  {key}: layers {[l['layer'] for l in sorted_layers]}")