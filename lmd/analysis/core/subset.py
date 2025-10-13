from typing import Optional
import numpy as np
import pandas as pd
from .types import SubsetSpec, RunData, RunInfo

def select_subset(labels: np.ndarray,
                  metadata: Optional[pd.DataFrame],
                  spec: SubsetSpec) -> np.ndarray:
    """Select subset of samples based on specification."""
    n = len(labels)
    idx_all = np.arange(n, dtype=np.int64)
    
    if spec.mode == "all":
        return idx_all
    
    rng = np.random.RandomState(spec.seed)
    
    if spec.frac is not None:
        size = max(1, min(n, int(round(spec.frac * n))))
    else:
        size = spec.size if spec.size is not None else n
    
    size = int(min(size, n))
    
    pos_idx = idx_all[labels == 1]
    neg_idx = idx_all[labels == 0]
    
    if spec.mode == "correct":
        if len(pos_idx) == 0:
            return np.array([], dtype=np.int64)
        choice = rng.choice(pos_idx, size=min(size, len(pos_idx)), replace=False)
        return np.sort(choice)
    
    if spec.mode == "incorrect":
        if len(neg_idx) == 0:
            return np.array([], dtype=np.int64)
        choice = rng.choice(neg_idx, size=min(size, len(neg_idx)), replace=False)
        return np.sort(choice)
    
    if spec.mode == "random":
        if spec.stratify_by and metadata is not None and spec.stratify_by in metadata.columns:
            groups = metadata[spec.stratify_by].unique()
            choices = []
            for group in groups:
                group_mask = (metadata[spec.stratify_by] == group).values
                group_idx = idx_all[group_mask]
                if len(group_idx) == 0:
                    continue
                group_size = int(round(size * len(group_idx) / n))
                if group_size > 0:
                    group_choice = rng.choice(group_idx, size=min(group_size, len(group_idx)), replace=False)
                    choices.append(group_choice)
            if choices:
                return np.sort(np.concatenate(choices))
            else:
                return np.array([], dtype=np.int64)
        
        choice = rng.choice(idx_all, size=size, replace=False)
        return np.sort(choice)
    
    if spec.mode == "balanced":
        p = 0.5
        n_pos = int(round(size * p))
        n_neg = size - n_pos
        pos_take = min(n_pos, len(pos_idx))
        neg_take = min(n_neg, len(neg_idx))
        choice = []
        if pos_take > 0:
            choice.append(rng.choice(pos_idx, size=pos_take, replace=False))
        if neg_take > 0:
            choice.append(rng.choice(neg_idx, size=neg_take, replace=False))
        if not choice:
            return np.array([], dtype=np.int64)
        return np.sort(np.concatenate(choice))
    
    return idx_all

def apply_subset_to_data(data: RunData, sel_idx: np.ndarray) -> RunData:
    """Apply subset selection to RunData."""
    labels = data.labels[sel_idx]
    rows = data.row_indices[sel_idx]
    
    n = len(labels)
    n_correct = int(labels.sum())
    n_incorrect = int(n - n_correct)
    
    info = RunInfo(
        n_samples=n,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        accuracy=float(labels.mean()) if n > 0 else 0.0,
        n_layers=data.info.n_layers,
        hidden_dim=data.info.hidden_dim,
        model=data.info.model,
        dataset=data.info.dataset,
        language=data.info.language,
        n_filtered=data.info.n_filtered
    )
    
    return RunData(
        arrays=data.arrays,
        metadata=data.metadata.iloc[sel_idx].reset_index(drop=True),
        outputs=data.outputs,
        correct_indices=rows[labels == 1],
        incorrect_indices=rows[labels == 0],
        info=info,
        labels=labels,
        row_indices=rows
    )