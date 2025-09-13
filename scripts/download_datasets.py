#!/usr/bin/env python
"""Download all datasets in a unified, clean JSONL format."""

import json
import jsonlines
import argparse
import re
from pathlib import Path
from datasets import load_dataset, get_dataset_config_names
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedDatasetDownloader:
    """Downloads and converts datasets to unified format"""
    
    def __init__(self, output_dir: str = "storage/datasets", max_samples: int = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
    
    def download_mgsm(self) -> List[Dict[str, Any]]:
        """Download MGSM multilingual math dataset (test split only)"""
        logger.info("Downloading MGSM dataset...")
        
        try:
            dataset = load_dataset("juletxara/mgsm", "en", split="test")
            
            data = []
            for idx, item in enumerate(dataset):
                if self.max_samples and idx >= self.max_samples:
                    break
                    
                unified = {
                    "id": idx,
                    "dataset": "mgsm",
                    "question": item["question"],
                    "answer": str(item["answer_number"]) if item["answer_number"] is not None else "",
                    "type": "math_word_problem",
                    "language": "en"
                }
                data.append(unified)
            
            logger.info(f"  Downloaded {len(data)} MGSM samples")
            return data
            
        except Exception as e:
            logger.error(f"  Failed to download MGSM: {e}")
            return []
    
    def download_gsm8k(self) -> List[Dict[str, Any]]:
        """Download GSM8K grade school math dataset (test split only)"""
        logger.info("Downloading GSM8K dataset...")
        
        try:
            dataset = load_dataset("gsm8k", "main", split="test")
            
            data = []
            for idx, item in enumerate(dataset):
                if self.max_samples and idx >= self.max_samples:
                    break
                
                # Extract numerical answer from the solution
                answer = item["answer"].split("####")[-1].strip()
                
                # Clean the solution by removing the #### answer part
                clean_solution = item["answer"].split("####")[0].strip()
                
                unified = {
                    "id": idx,
                    "dataset": "gsm8k",
                    "question": item["question"],
                    "answer": answer,
                    "type": "math_word_problem",
                    "full_solution": clean_solution
                }
                data.append(unified)
            
            logger.info(f"  Downloaded {len(data)} GSM8K samples")
            return data
            
        except Exception as e:
            logger.error(f"  Failed to download GSM8K: {e}")
            return []
    
    def download_math(self) -> List[Dict[str, Any]]:
        """Download MATH competition mathematics dataset (test split only)"""
        logger.info("Downloading MATH dataset...")
        
        # Try multiple sources for MATH dataset
        math_sources = [
            ("EleutherAI/hendrycks_math", "all"),
            ("lighteval/MATH-Hard", "test"),
            ("hendrycks/competition_math", "test")
        ]
        
        for source, config in math_sources:
            try:
                logger.info(f"  Trying source: {source}")
                if config == "all":
                    # Get all available configs
                    configs = get_dataset_config_names(source)
                    logger.info(f"    Available configs: {configs[:5]}...")  # Show first 5
                    
                    data = []
                    for cfg in configs:
                        if self.max_samples and len(data) >= self.max_samples:
                            break
                        try:
                            dataset = load_dataset(source, cfg, split="test")
                            for idx, item in enumerate(dataset):
                                if self.max_samples and len(data) >= self.max_samples:
                                    break
                                
                                answer = self._extract_math_answer(item.get("solution", ""))
                                
                                unified = {
                                    "id": f"{cfg}_{idx}",
                                    "dataset": "math",
                                    "question": item.get("problem", ""),
                                    "answer": answer,
                                    "type": cfg,
                                    "level": item.get("level", ""),
                                    "solution": item.get("solution", "")
                                }
                                data.append(unified)
                        except Exception as e:
                            logger.debug(f"    Config {cfg} failed: {e}")
                            continue
                else:
                    dataset = load_dataset(source, config, split="test")
                    
                    data = []
                    for idx, item in enumerate(dataset):
                        if self.max_samples and idx >= self.max_samples:
                            break
                        
                        answer = self._extract_math_answer(item.get("solution", ""))
                        
                        unified = {
                            "id": idx,
                            "dataset": "math",
                            "question": item.get("problem", ""),
                            "answer": answer,
                            "type": item.get("type", "unknown"),
                            "level": item.get("level", ""),
                            "solution": item.get("solution", "")
                        }
                        data.append(unified)
                
                if data:
                    logger.info(f"  Downloaded {len(data)} MATH samples from {source}")
                    return data
                    
            except Exception as e:
                logger.warning(f"  Source {source} failed: {e}")
                continue
        
        logger.error("  All MATH sources failed")
        return []
    
    def download_commonsenseqa(self) -> List[Dict[str, Any]]:
        """Download CommonsenseQA dataset (validation split as test)"""
        logger.info("Downloading CommonsenseQA dataset...")
        
        try:
            # Use validation split as test split has no labels
            dataset = load_dataset("commonsense_qa", split="validation")
            
            data = []
            for idx, item in enumerate(dataset):
                if self.max_samples and idx >= self.max_samples:
                    break
                
                # Format question with choices
                choices = item["choices"]
                question_text = item["question"]
                
                # Create formatted question with options
                options = "\n".join([
                    f"{label}. {text}"
                    for label, text in zip(choices["label"], choices["text"])
                ])
                
                full_question = f"{question_text}\n\nChoices:\n{options}"
                
                unified = {
                    "id": idx,
                    "dataset": "commonsenseqa",
                    "question": full_question,
                    "answer": item["answerKey"],
                    "type": "multiple_choice",
                    "num_choices": len(choices["label"])
                }
                data.append(unified)
            
            logger.info(f"  Downloaded {len(data)} CommonsenseQA samples")
            return data
            
        except Exception as e:
            logger.error(f"  Failed to download CommonsenseQA: {e}")
            return []
    
    def download_hotpotqa(self) -> List[Dict[str, Any]]:
        """Download HotpotQA multi-hop reasoning dataset (validation split only)"""
        logger.info("Downloading HotpotQA dataset...")
        
        try:
            # Use labeled dev/validation; 'distractor' config is most commonly used
            dataset = load_dataset("hotpot_qa", "distractor", split="validation")
            
            data = []
            for idx, item in enumerate(dataset):
                if self.max_samples and idx >= self.max_samples:
                    break
                
                # Format the question with context
                question = item["question"]
                context = item.get("context", {})
                
                # Create context text from context dictionary
                context_parts = []
                if context and "title" in context and "sentences" in context:
                    titles = context["title"]
                    sentences_list = context["sentences"]
                    
                    for i, (title, sentences) in enumerate(zip(titles, sentences_list)):
                        if i < 5:  # Limit to first 5 contexts to avoid too long prompts
                            context_text = f"{title}: " + " ".join(sentences)
                            context_parts.append(context_text)
                
                # Combine question with context
                if context_parts:
                    context_text = "\n\n".join(context_parts)
                    full_question = f"{question}\n\nContext:\n{context_text}"
                else:
                    full_question = question
                
                unified = {
                    "id": idx,
                    "dataset": "hotpotqa",
                    "question": full_question,
                    "answer": item["answer"],
                    "type": "multi_hop_reasoning",
                    "level": item.get("level", "medium"),
                    "supporting_facts": item.get("supporting_facts", [])
                }
                
                data.append(unified)
            
            logger.info(f"  Downloaded {len(data)} HotpotQA samples")
            return data
            
        except Exception as e:
            logger.error(f"  Failed to download HotpotQA: {e}")
            return []
    
    def download_theoremqa(self) -> List[Dict[str, Any]]:
        """Download TheoremQA dataset (test split only)"""
        logger.info("Downloading TheoremQA dataset...")
        
        try:
            dataset = load_dataset("TIGER-Lab/TheoremQA", split="test")
            
            data = []
            for idx, item in enumerate(dataset):
                if self.max_samples and idx >= self.max_samples:
                    break
                
                unified = {
                    "id": idx,
                    "dataset": "theoremqa",
                    "question": item["Question"],
                    "answer": str(item["Answer"]),
                    "answer_type": item.get("Answer_type", ""),
                    "type": "theorem_application"
                }
                
                # Add theorem field if available
                if "Theorem" in item:
                    unified["theorem"] = item["Theorem"]
                
                data.append(unified)
            
            logger.info(f"  Downloaded {len(data)} TheoremQA samples")
            return data
            
        except Exception as e:
            logger.error(f"  Failed to download TheoremQA: {e}")
            return []
    
    def download_belebele(self) -> List[Dict[str, Any]]:
        """Download Belebele multilingual reading comprehension dataset"""
        logger.info("Downloading Belebele dataset...")
        
        try:
            # Load English configuration directly
            dataset = load_dataset("facebook/belebele", "eng_Latn", split="test")
            
            data = []
            for idx, item in enumerate(dataset):
                if self.max_samples and idx >= self.max_samples:
                    break
                    
                # Format question and options
                question = item["question"]
                passage = item["flores_passage"]
                
                # Build options
                options = []
                for i in range(1, 5):  # mc_answer1 to mc_answer4
                    option_key = f"mc_answer{i}"
                    if option_key in item:
                        options.append(f"({chr(64+i)}) {item[option_key]}")
                
                # Combine into full question
                full_question = f"Passage:\n{passage}\n\nQuestion: {question}\n\nChoices:\n" + "\n".join(options)
                
                # Convert answer number to letter (handle both string and int)
                answer_num = item.get("correct_answer_num", 0)
                if isinstance(answer_num, str):
                    answer_num = int(answer_num) if answer_num.isdigit() else 0
                answer = chr(64 + answer_num) if answer_num > 0 else ""
                
                unified = {
                    "id": idx,
                    "dataset": "belebele",
                    "en": full_question,  # Use 'en' key as shown in their example
                    "answer": answer,
                    "type": "reading_comprehension",
                    "language": "en"
                }
                data.append(unified)
            
            logger.info(f"  Downloaded {len(data)} Belebele samples")
            return data
            
        except Exception as e:
            logger.error(f"  Failed to download Belebele: {e}")
            return []
    
    def download_mmlu_complete(self) -> List[Dict[str, Any]]:
        """Download complete MMLU dataset (all 57 subjects, test split only)"""
        logger.info("Downloading MMLU dataset (all subjects)...")
        
        try:
            # Use the 'all' config to get all 57 subjects at once
            dataset = load_dataset("cais/mmlu", "all", split="test")
            
            all_data = []
            for idx, item in enumerate(dataset):
                if self.max_samples and idx >= self.max_samples:
                    break
                
                # Format multiple choice question
                question = item["question"]
                choices = item["choices"]
                
                options = "\n".join([
                    f"{chr(65+i)}. {choice}"
                    for i, choice in enumerate(choices)
                ])
                
                full_question = f"{question}\n\nChoices:\n{options}"
                
                unified = {
                    "id": idx,
                    "dataset": "mmlu",
                    "question": full_question,
                    "answer": chr(65 + item["answer"]),  # Convert 0->A, 1->B, etc.
                    "type": "multiple_choice",
                    "subject": item.get("subject", "unknown"),
                    "num_choices": len(choices)
                }
                all_data.append(unified)
            
            logger.info(f"  Downloaded {len(all_data)} MMLU samples")
            return all_data
            
        except Exception as e:
            logger.error(f"  Failed to download MMLU: {e}")
            return []
    
    def _extract_math_answer(self, solution: str) -> str:
        """Extract answer from MATH solution text"""
        if not solution:
            return ""
        
        # Try to find \boxed{...} pattern
        boxed_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = re.findall(boxed_pattern, solution, re.DOTALL)
        
        if matches:
            # Take the last boxed answer
            answer = matches[-1]
            # Clean up the answer
            answer = answer.strip()
            # Remove dollar signs if present
            answer = answer.replace("$", "")
            return answer
        
        # Fallback: look for "The answer is"
        if "The answer is" in solution:
            answer = solution.split("The answer is")[-1]
            # Take until period or newline
            answer = answer.split(".")[0].split("\n")[0].strip()
            return answer
        
        # If no clear answer found, return empty
        return ""
    
    def save_dataset(self, data: List[Dict[str, Any]], filename: str):
        """Save dataset to JSONL file"""
        output_path = self.output_dir / filename
        
        with jsonlines.open(output_path, mode='w') as writer:
            for item in data:
                writer.write(item)
        
        logger.info(f"  Saved {len(data)} samples to {output_path}")
    
    def download_all(self):
        """Download all datasets"""
        datasets = {
            "mgsm.jsonl": self.download_mgsm,
            "gsm8k.jsonl": self.download_gsm8k,
            "math.jsonl": self.download_math,
            "commonsenseqa.jsonl": self.download_commonsenseqa,
            "hotpotqa.jsonl": self.download_hotpotqa,
            "theoremqa.jsonl": self.download_theoremqa,
            "mmlu.jsonl": self.download_mmlu_complete
        }
        
        for filename, download_func in datasets.items():
            try:
                data = download_func()
                if data:
                    self.save_dataset(data, filename)
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")
                continue
        
        logger.info("All downloads complete!")
        self._print_summary()
    
    def _print_summary(self):
        """Print summary of downloaded datasets"""
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        
        total_samples = 0
        for jsonl_file in sorted(self.output_dir.glob("*.jsonl")):
            with jsonlines.open(jsonl_file) as reader:
                count = sum(1 for _ in reader)
                print(f"{jsonl_file.name:20s}: {count:6d} samples")
                total_samples += count
        
        print("-"*60)
        print(f"{'TOTAL':20s}: {total_samples:6d} samples")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Download all datasets in unified format")
    parser.add_argument("--output-dir", type=str, default="storage/datasets",
                       help="Output directory for datasets")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per dataset (for testing)")
    parser.add_argument("--dataset", type=str, default="all",
                       choices=["all", "mgsm", "gsm8k", "math", 
                               "commonsenseqa", "hotpotqa", "theoremqa", "mmlu", "belebele"],
                       help="Specific dataset to download")
    args = parser.parse_args()
    
    downloader = UnifiedDatasetDownloader(
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )
    
    if args.dataset == "all":
        downloader.download_all()
    else:
        # Download specific dataset
        dataset_map = {
            "mgsm": (downloader.download_mgsm, "mgsm.jsonl"),
            "gsm8k": (downloader.download_gsm8k, "gsm8k.jsonl"),
            "math": (downloader.download_math, "math.jsonl"),
            "commonsenseqa": (downloader.download_commonsenseqa, "commonsenseqa.jsonl"),
            "hotpotqa": (downloader.download_hotpotqa, "hotpotqa.jsonl"),
            "theoremqa": (downloader.download_theoremqa, "theoremqa.jsonl"),
            "mmlu": (downloader.download_mmlu_complete, "mmlu.jsonl"),
            "belebele": (downloader.download_belebele, "belebele.jsonl")
        }
        
        if args.dataset in dataset_map:
            func, filename = dataset_map[args.dataset]
            data = func()
            if data:
                downloader.save_dataset(data, filename)


if __name__ == "__main__":
    main()