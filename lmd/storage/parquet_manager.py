"""Parquet storage management for metadata and outputs."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from ..core.data_processor import ProcessedSample


class ParquetManager:
    """Manages Parquet storage for metadata and outputs."""
    
    def __init__(self, output_dir: Path):
        """Initialize Parquet manager."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _merge_and_write(self, path: Path, new_df: pd.DataFrame, key_cols=["sample_id"]):
        """Merge and write Parquet file, avoiding overwriting historical data"""
        if path.exists():
            old = pd.read_parquet(path)
            merged = pd.concat([old, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=key_cols, keep="last")
        else:
            merged = new_df
        merged.to_parquet(path, index=False)
    
    def save_metadata(self, processed_samples: List[ProcessedSample]) -> None:
        """Save metadata to Parquet file."""
        metadata_rows = []
        
        for sample in processed_samples:
            metadata_rows.append({
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "language": sample.language,
                "model": getattr(sample, 'model', 'unknown'),
                "question": sample.question,
                "answer_gt": sample.answer_gt,
                "answer_type": sample.answer_type,
                "is_correct": sample.is_correct,
                "answer_token_count": sample.hidden_state_data.answer_token_count,
                "finish_reason": sample.finish_reason,
                "generated_length": len(sample.answer_token_ids),
                "prompt_length": sample.input_length,
                "extracted_answer": sample.extracted_answer
            })
        
        df_new = pd.DataFrame(metadata_rows)
        self._merge_and_write(self.output_dir / "metadata.parquet", df_new, key_cols=["sample_id"])
    
    def save_outputs(self, processed_samples: List[ProcessedSample]) -> None:
        """Save outputs to Parquet file."""
        output_rows = []
        
        for sample in processed_samples:
            output_rows.append({
                "sample_id": sample.sample_id,
                "prompt": sample.prompt,
                "generated_text": sample.generated_text,
                "generated_text_raw": sample.generated_text_raw,
                "extracted_answer": sample.extracted_answer,
                "finish_reason": sample.finish_reason,
                "answer_token_ids": sample.answer_token_ids,
                "max_probability": sample.generation_metrics.max_probability,
                "perplexity": sample.generation_metrics.perplexity,
                "entropy": sample.generation_metrics.entropy
            })
        
        df_new = pd.DataFrame(output_rows)
        self._merge_and_write(self.output_dir / "outputs.parquet", df_new, key_cols=["sample_id"])
    
    def load_metadata(self) -> pd.DataFrame:
        """Load metadata from Parquet file."""
        return pd.read_parquet(self.output_dir / "metadata.parquet")
    
    def load_outputs(self) -> pd.DataFrame:
        """Load outputs from Parquet file."""
        return pd.read_parquet(self.output_dir / "outputs.parquet")
    
    def filter_samples(self, 
                      where_clause: str = None,
                      limit: int = None) -> np.ndarray:
        """Filter samples based on conditions."""
        df = self.load_metadata()
        
        if where_clause:
            # Simple filtering - can be extended for complex queries
            if "correct" in where_clause:
                if "true" in where_clause.lower():
                    df = df[df["is_correct"] == True]
                elif "false" in where_clause.lower():
                    df = df[df["is_correct"] == False]
        
        if limit:
            df = df.head(limit)
        
        return df["sample_id"].values
