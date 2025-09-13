"""Parquet storage management for metadata and outputs."""

import pandas as pd
from pathlib import Path
from typing import List


class ParquetManager:
    """Manages Parquet storage for metadata and outputs."""
    
    def __init__(self, output_dir: Path):
        """Initialize Parquet manager."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_metadata(self, processed_samples: List) -> None:
        """Save metadata to Parquet file."""
        metadata_rows = []
        
        for sample in processed_samples:
            metadata_rows.append({
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "language": sample.language,
                "model": sample.model,
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
        
        df = pd.DataFrame(metadata_rows)
        df.to_parquet(self.output_dir / "metadata.parquet", index=False)
    
    def save_outputs(self, processed_samples: List) -> None:
        """Save outputs to Parquet file."""
        output_rows = []
        
        for sample in processed_samples:
            output_rows.append({
                "sample_id": sample.sample_id,
                "prompt": sample.prompt,
                "generated_text": sample.generated_text,
                "extracted_answer": sample.extracted_answer,
                "max_probability": sample.generation_metrics.max_probability,
                "perplexity": sample.generation_metrics.perplexity,
                "entropy": sample.generation_metrics.entropy
            })
        
        df = pd.DataFrame(output_rows)
        df.to_parquet(self.output_dir / "outputs.parquet", index=False)