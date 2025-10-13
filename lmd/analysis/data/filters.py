import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from ..core.types import FilterSpec
from ..config import LABEL_CORRECT, LABEL_INCORRECT

def build_sample_mapping(arrays: Dict[str, Any]) -> Dict[int, int]:
    sample_ids = np.asarray(arrays["sample_id"])
    sample_valid = np.asarray(arrays["sample_valid"], dtype=bool)
    return {
        int(sid): int(i) 
        for i, sid in enumerate(sample_ids) 
        if sample_valid[i] and sid >= 0
    }

def apply_filters(metadata: pd.DataFrame,
                  sample_mapping: Dict[int, int],
                  spec: FilterSpec) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], pd.DataFrame]:
    initial_count = len(metadata)
    df = metadata.copy()
    stats = {'initial': initial_count}
    
    if spec.exclude_extraction_failures and "extracted_answer" in df.columns:
        before = len(df)
        df = df[df["extracted_answer"].astype(str).str.strip() != ""]
        df = df[~df["extracted_answer"].isna()]
        stats['extraction_failures_removed'] = before - len(df)
    
    if spec.exclude_truncated and "finish_reason" in df.columns:
        before = len(df)
        df = df[df["finish_reason"] != "length"]
        stats['length_truncated_removed'] = before - len(df)
    
    if spec.min_answer_tokens and "answer_token_count" in df.columns:
        before = len(df)
        df = df[df["answer_token_count"] >= spec.min_answer_tokens]
        stats[f'below_{spec.min_answer_tokens}_tokens_removed'] = before - len(df)
    
    if spec.max_answer_tokens and "answer_token_count" in df.columns:
        before = len(df)
        df = df[df["answer_token_count"] <= spec.max_answer_tokens]
        stats[f'above_{spec.max_answer_tokens}_tokens_removed'] = before - len(df)
    
    if spec.require_valid_in_zarr:
        before = len(df)
        df = df[df["sample_id"].isin(sample_mapping.keys())]
        stats['not_in_zarr_removed'] = before - len(df)
    
    stats['final'] = len(df)
    stats['filtered_out'] = initial_count - len(df)
    
    df = df.copy()
    df['zarr_row'] = df['sample_id'].map(sample_mapping).astype(np.int64)
    
    indices = df['zarr_row'].to_numpy()
    labels = df['is_correct'].astype(int).to_numpy()
    
    if spec.verbose:
        print(f"Filtering: {stats['final']}/{stats['initial']} samples kept")
        for key, value in stats.items():
            if 'removed' in key:
                print(f"  {key}: {value}")
    
    return indices, labels, stats, df