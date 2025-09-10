"""Dataset registry and management utilities."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import jsonlines
import pandas as pd

from ..utils.types import SampleRecord


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    file_path: Path
    sample_count: int
    supported_languages: List[str]
    answer_types: List[str]


class DatasetRegistry:
    """Registry for managing datasets."""
    
    def __init__(self, data_root: Path):
        """Initialize dataset registry."""
        self.data_root = Path(data_root)
        self._datasets: Dict[str, DatasetInfo] = {}
        self._load_datasets()
    
    def _load_datasets(self) -> None:
        """Load available datasets from data directory."""
        if not self.data_root.exists():
            return
        
        for jsonl_file in self.data_root.glob("*.jsonl"):
            dataset_name = jsonl_file.stem
            sample_count = self._count_samples(jsonl_file)
            
            self._datasets[dataset_name] = DatasetInfo(
                name=dataset_name,
                file_path=jsonl_file,
                sample_count=sample_count,
                supported_languages=self._detect_languages(jsonl_file),
                answer_types=self._detect_answer_types(jsonl_file)
            )
    
    def _count_samples(self, file_path: Path) -> int:
        """Count samples in a JSONL file."""
        try:
            with jsonlines.open(file_path) as reader:
                return sum(1 for _ in reader)
        except Exception:
            return 0
    
    def _detect_languages(self, file_path: Path) -> List[str]:
        """Detect supported languages in dataset."""
        languages = set()
        try:
            with jsonlines.open(file_path) as reader:
                for item in reader:
                    for key in item.keys():
                        if key in ["en", "bn", "de", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]:
                            languages.add(key)
        except Exception:
            pass
        return list(languages) or ["en"]
    
    def _detect_answer_types(self, file_path: Path) -> List[str]:
        """Detect answer types in dataset."""
        answer_types = set()
        try:
            with jsonlines.open(file_path) as reader:
                for item in reader:
                    if "answer_type" in item:
                        answer_types.add(item["answer_type"])
        except Exception:
            pass
        return list(answer_types)
    
    def load_samples(self, 
                    dataset_name: str, 
                    language: str = "en",
                    max_samples: Optional[int] = None) -> List[SampleRecord]:
        """Load samples from dataset."""
        if dataset_name not in self._datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset_info = self._datasets[dataset_name]
        samples = []
        
        with jsonlines.open(dataset_info.file_path) as reader:
            for idx, item in enumerate(reader):
                if max_samples and idx >= max_samples:
                    break
                
                sample = self._parse_sample(item, dataset_name, language, idx)
                if sample:
                    samples.append(sample)
        
        return samples
    
    def _parse_sample(self, item: Dict[str, Any], dataset: str, language: str, idx: int) -> Optional[SampleRecord]:
        """Parse a single sample from raw data."""
        try:
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
        except Exception:
            return None
    
    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetInfo]:
        """Get information about a dataset."""
        return self._datasets.get(dataset_name)
    
    def list_datasets(self) -> List[str]:
        """List available datasets."""
        return list(self._datasets.keys())
    
    def get_sample_count(self, dataset_name: str) -> int:
        """Get sample count for dataset."""
        if dataset_name in self._datasets:
            return self._datasets[dataset_name].sample_count
        return 0
