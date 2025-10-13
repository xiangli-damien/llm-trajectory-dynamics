import zarr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from numcodecs import blosc
from ..core.types import RunConfig, FilterSpec, RunInfo, RunData
from ..config import NUMCODECS_THREADS, DEFAULT_CACHE_SIZE_GB
from .filters import apply_filters, build_sample_mapping

blosc.set_nthreads(NUMCODECS_THREADS)

def open_run(config: RunConfig, cache_size_gb: float = DEFAULT_CACHE_SIZE_GB) -> Dict[str, Any]:
    zarr_path = config.run_dir / "zarr" / f"{config.model}__{config.dataset}__{config.language}"
    
    parquet_path_with_lang = config.run_dir / "parquet" / config.model / config.dataset / config.language
    parquet_path_without_lang = config.run_dir / "parquet" / config.model / config.dataset
    
    if parquet_path_with_lang.exists():
        parquet_path = parquet_path_with_lang
    elif parquet_path_without_lang.exists():
        parquet_path = parquet_path_without_lang
    else:
        parquet_path = parquet_path_with_lang
    
    store = zarr.DirectoryStore(str(zarr_path))
    cache = zarr.LRUStoreCache(store, max_size=int(cache_size_gb * (1024**3)))
    zg = zarr.open_group(cache, mode="r")
    
    arrays = {
        "mean_answer_hs": zg["mean_answer_hs"],
        "prompt_last_hs": zg.get("prompt_last_hs"),
        "answer_tok_values": zg.get("answer_tok_values"),
        "answer_tok_ptr": zg["answer_tok_ptr"][:] if "answer_tok_ptr" in zg else None,
        "sample_id": zg["sample_id"][:],
        "sample_valid": zg["sample_valid"][:].astype(bool) if "sample_valid" in zg else np.ones(len(zg["sample_id"][:]), dtype=bool),
    }
    
    metadata = pd.read_parquet(parquet_path / "metadata.parquet")
    outputs_path = parquet_path / "outputs.parquet"
    outputs = pd.read_parquet(outputs_path) if outputs_path.exists() else pd.DataFrame()
    
    return {"arrays": arrays, "metadata": metadata, "outputs": outputs}

def load_run_data(config: RunConfig,
                  filter_spec: Optional[FilterSpec] = None,
                  cache_size_gb: float = DEFAULT_CACHE_SIZE_GB) -> RunData:
    data = open_run(config, cache_size_gb)
    arrays = data["arrays"]
    metadata = data["metadata"]
    outputs = data["outputs"]
    
    sample_mapping = build_sample_mapping(arrays)
    
    if filter_spec is None:
        filter_spec = FilterSpec()
    
    indices, labels, stats, metadata_filtered = apply_filters(
        metadata, sample_mapping, filter_spec
    )
    
    correct_indices = indices[labels == 1]
    incorrect_indices = indices[labels == 0]
    
    n_layers = int(arrays["mean_answer_hs"].shape[1])
    hidden_dim = int(arrays["mean_answer_hs"].shape[2])
    
    info = RunInfo(
        n_samples=len(indices),
        n_correct=len(correct_indices),
        n_incorrect=len(incorrect_indices),
        accuracy=float(labels.mean()) if len(labels) > 0 else 0.0,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        model=config.model,
        dataset=config.dataset,
        language=config.language,
        n_filtered=stats.get('filtered_out', 0)
    )
    
    return RunData(
        arrays=arrays,
        metadata=metadata_filtered,
        outputs=outputs,
        correct_indices=correct_indices,
        incorrect_indices=incorrect_indices,
        info=info,
        labels=labels,
        row_indices=indices
    )