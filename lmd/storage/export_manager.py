"""Export utilities for different formats."""

import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from ..core.data_processor import ProcessedSample
from .zarr_manager import ZarrManager


class ExportManager:
    """Manages export to different formats."""
    
    def __init__(self, output_dir: Path):
        """Initialize export manager."""
        self.output_dir = Path(output_dir)
    
    
    def export_analysis_format(self,
                              processed_samples: List[ProcessedSample],
                              output_file: str = "analysis_data.pkl") -> None:
        """Export data in analysis-friendly format."""
        analysis_data = {
            'samples': processed_samples,
            'summary': {
                'total_samples': len(processed_samples),
                'correct_samples': sum(1 for s in processed_samples if s.is_correct),
                'accuracy': sum(1 for s in processed_samples if s.is_correct) / len(processed_samples) if processed_samples else 0.0,
                'avg_answer_tokens': np.mean([s.hidden_state_data.answer_token_count for s in processed_samples]) if processed_samples else 0.0
            }
        }
        
        output_path = self.output_dir / output_file
        with open(output_path, 'wb') as f:
            pickle.dump(analysis_data, f)
        
        print(f"Analysis data exported to: {output_path}")
