"""Dataset registry and management utilities."""

from typing import List, Optional
from pathlib import Path
import jsonlines

from ..utils.types import SampleRecord


class DatasetRegistry:
    """Registry for managing datasets."""
    
    def __init__(self, data_root: Path):
        """Initialize dataset registry."""
        self.data_root = Path(data_root)
    
    def load_samples(self, 
                    dataset_name: str, 
                    language: str = "en",
                    max_samples: Optional[int] = None) -> List[SampleRecord]:
        """Load samples from dataset."""
        file_path = self.data_root / f"{dataset_name}.jsonl"
        if not file_path.exists():
            raise ValueError(f"Dataset file not found: {file_path}")
        
        samples = []
        with jsonlines.open(file_path) as reader:
            for idx, item in enumerate(reader):
                if max_samples and idx >= max_samples:
                    break
                
                sample = self._parse_sample(item, dataset_name, language, idx)
                if sample:
                    samples.append(sample)
        
        return samples
    
    def _parse_sample(self, item: dict, dataset: str, language: str, idx: int) -> Optional[SampleRecord]:
        """Parse a single sample from raw data."""
        # Extract question based on language
        if language in item:
            question = item[language]
        elif "question" in item:
            question = item["question"]
        elif "en" in item:
            question = item["en"]
        else:
            return None
        
        # Extract answer
        answer_gt = str(item.get("answer", ""))
        if not answer_gt:
            return None
        
        return SampleRecord(
            sample_id=idx,
            question=question,
            answer_gt=answer_gt,
            dataset=dataset,
            language=language,
            answer_type=item.get("answer_type")
        )