"""Answer parsing and validation utilities."""

import re
import unicodedata
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..utils.types import SampleRecord


@dataclass
class ParseResult:
    """Result of answer parsing."""
    extracted_answer: str
    is_correct: bool
    confidence: float = 1.0


class AnswerParser:
    """Robust answer extraction and validation for multiple datasets."""
    
    # Unified patterns
    NUMBER_PATTERN = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    
    # Language-specific answer prefixes
    ANSWER_PREFIXES = {
        "en": "Answer",
        "bn": "উত্তর",
        "de": "Antwort", 
        "es": "Respuesta",
        "fr": "Réponse",
        "ja": "答え",
        "ru": "Ответ",
        "sw": "Jibu",
        "te": "సమాధానం",
        "th": "คำตอบ",
        "zh": "答案",
    }
    
    def __init__(self, dataset: str):
        """Initialize answer parser with pre-compiled patterns."""
        self.dataset = dataset.lower()
        
        # Pre-compile common patterns
        self._number_re = re.compile(self.NUMBER_PATTERN)
        self._answer_is_re = re.compile(
            r'(?:Therefore,?\s*)?(?:the\s+)?answer\s+is\s*(.+?)(?:(?<!\d)\.(?!\d)|$)',
            re.IGNORECASE | re.DOTALL
        )
        self._boxed_re = re.compile(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', re.DOTALL)
        self._frac_re = re.compile(r'\\frac\{([^}]+)\}\{([^}]+)\}')
        
        # Multiple choice patterns
        self._mc4_re = re.compile(r'Answer\s*:\s*([A-D])', re.IGNORECASE)
        self._mc5_re = re.compile(r'Answer\s*:\s*([A-E])', re.IGNORECASE)
        
        # Dataset-specific extractors
        self._extractors = {
            "mgsm": self._extract_numerical,
            "gsm8k": self._extract_numerical,
            "math": self._extract_math,
            "mmlu": lambda t, s: self._extract_mc(t, 4),
            "belebele": lambda t, s: self._extract_mc(t, 4),
            "commonsenseqa": lambda t, s: self._extract_mc(t, 5),
            "hotpotqa": self._extract_hotpot,
            "theoremqa": self._extract_theorem,
        }
    
    def parse_and_label(self, generated_text: str, sample: SampleRecord) -> Dict[str, Any]:
        """Parse generated text and validate correctness."""
        text = self._normalize_text(generated_text)
        extracted = self._extract_answer(text, sample)
        is_correct = self._validate_answer(extracted, sample.answer_gt, sample)
        
        return {
            "extracted_answer": extracted,
            "correct": is_correct
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize unicode and common punctuation."""
        if not text:
            return ""
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        # Normalize common punctuation
        return text.replace("．", ".").replace("。", ".").replace("：", ":")
    
    def _extract_answer(self, text: str, sample: SampleRecord) -> str:
        """Extract answer using dataset-specific logic."""
        extractor = self._extractors.get(self.dataset, self._extract_generic)
        return extractor(text, sample)
    
    def _extract_numerical(self, text: str, sample: SampleRecord) -> str:
        """Extract numerical answer with language support."""
        prefix = self.ANSWER_PREFIXES.get(sample.language, "Answer")
        
        # Try structured patterns first
        patterns = [
            rf'(?:^|\n)\s*{re.escape(prefix)}:\s*(.+?)(?:(?<!\d)\.(?!\d)|$)',
            r'(?:^|\n)\s*Answer:\s*(.+?)(?:(?<!\d)\.(?!\d)|$)',
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
            if matches:
                candidate = matches[-1].group(1).strip().replace(",", "")
                nums = self._number_re.findall(candidate)
                if nums:
                    return nums[-1].rstrip(".")
        
        # Try "answer is" pattern
        match = self._answer_is_re.search(text)
        if match:
            candidate = match.group(1).strip().replace(",", "")
            nums = self._number_re.findall(candidate)
            if nums:
                return nums[-1].rstrip(".")
        
        # Fallback: last number in text
        nums = self._number_re.findall(text.replace(",", ""))
        return nums[-1].rstrip(".") if nums else ""
    
    def _extract_math(self, text: str, sample: SampleRecord) -> str:
        """Extract math answer from boxed format."""
        match = self._boxed_re.search(text)
        if not match:
            return ""
        
        answer = match.group(1).strip()
        # Clean nested braces
        while answer.startswith("{") and answer.endswith("}"):
            answer = answer[1:-1].strip()
        # Remove text content
        answer = re.sub(r'\\text\{[^}]*\}', '', answer)
        return answer
    
    def _extract_mc(self, text: str, n_choices: int) -> str:
        """Extract multiple choice answer."""
        pattern = self._mc5_re if n_choices == 5 else self._mc4_re
        matches = list(pattern.finditer(text))
        return matches[-1].group(1).upper() if matches else ""
    
    def _extract_hotpot(self, text: str, sample: SampleRecord) -> str:
        """Extract HotpotQA answer."""
        patterns = [
            r'Answer:\s*(.+?)(?:\.|$)',
            r'answer is\s*(.+?)(?:\.|$)',
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                return matches[-1].group(1).strip().rstrip(".")
        return ""
    
    def _extract_theorem(self, text: str, sample: SampleRecord) -> str:
        """Extract TheoremQA answer based on type."""
        answer_type = sample.answer_type
        
        # Try boxed format first
        match = self._boxed_re.search(text)
        candidate = match.group(1).strip() if match else None
        
        # Try "answer is" pattern
        if not candidate:
            match = self._answer_is_re.search(text)
            candidate = match.group(1).strip() if match else ""
        
        if not candidate:
            # Fallback to last number
            nums = self._number_re.findall(text.replace(",", ""))
            return nums[-1].rstrip(".") if nums else ""
        
        # Clean and process based on type
        candidate = self._clean_latex(candidate)
        
        if answer_type == "bool":
            low = candidate.lower()
            if any(x in low for x in ["true", "yes"]):
                return "True"
            if any(x in low for x in ["false", "no"]):
                return "False"
            return ""
        
        if answer_type in ["integer", "float"]:
            nums = self._number_re.findall(candidate.replace(",", ""))
            return nums[-1].rstrip(".") if nums else ""
        
        if answer_type in ["list of integer", "list of float"]:
            match = re.search(r'\[([^\]]+)\]', candidate)
            return match.group(1).strip() if match else ""
        
        if answer_type == "option":
            match = re.search(r'\(([a-d])\)', candidate, re.IGNORECASE)
            return match.group(1).upper() if match else ""
        
        return candidate
    
    def _extract_generic(self, text: str, sample: SampleRecord) -> str:
        """Generic answer extraction."""
        match = self._answer_is_re.search(text)
        if match:
            return match.group(1).strip()
        
        match = re.search(r'Answer:\s*(.+?)(?:\.|$)', text, re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def _clean_latex(self, text: str) -> str:
        """Clean LaTeX expressions."""
        text = text.replace("$", "")
        
        # Convert fractions
        match = self._frac_re.search(text)
        if match:
            try:
                num, den = float(match.group(1)), float(match.group(2))
                text = text.replace(match.group(0), str(num / den))
            except:
                pass
        
        # Remove LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = text.replace("\\(", "").replace("\\)", "")
        text = text.replace("{", "").replace("}", "")
        
        # Handle pi
        text = re.sub(r'(\d+(?:\.\d+)?)[\\]?pi', r'\1*3.14159', text)
        text = text.replace(r'\pi', '3.14159').replace('pi', '3.14159')
        
        # Try simple arithmetic
        if '*' in text and all(c in '0123456789.*+-' for c in text.replace(' ', '')):
            try:
                result = eval(text)
                if isinstance(result, (int, float)):
                    text = str(result)
            except:
                pass
        
        return text.strip()
    
    def _validate_answer(self, extracted: str, ground_truth: str, sample: SampleRecord) -> bool:
        """Validate extracted answer against ground truth."""
        if not extracted:
            return False
        
        # Normalize both answers
        extracted_norm = self._normalize_answer(extracted)
        gt_norm = self._normalize_answer(ground_truth)
        
        # Numerical validation for appropriate datasets
        if self.dataset in ["mgsm", "gsm8k", "math"] or \
           (self.dataset == "theoremqa" and sample.answer_type in ["integer", "float"]):
            if self._validate_numerical(extracted_norm, gt_norm):
                return True
            if self.dataset == "math":
                return extracted_norm.lower() == gt_norm.lower()
        
        # HotpotQA flexible matching
        if self.dataset == "hotpotqa":
            return self._validate_hotpot(extracted_norm, gt_norm)
        
        # Default string comparison
        return extracted_norm.lower() == gt_norm.lower()
    
    def _validate_numerical(self, extracted: str, ground_truth: str) -> bool:
        """Validate numerical answers with tolerance."""
        try:
            e_val = float(extracted)
            g_val = float(ground_truth)
            return abs(e_val - g_val) < 1e-6
        except:
            return False
    
    def _validate_hotpot(self, extracted: str, ground_truth: str) -> bool:
        """Validate HotpotQA with flexible matching."""
        e_low = extracted.lower().strip()
        g_low = ground_truth.lower().strip()
        
        # Exact or substring match
        if e_low == g_low or g_low in e_low or e_low in g_low:
            return True
        
        # Yes/No variations
        yes_vars = ["yes", "true", "correct"]
        no_vars = ["no", "false", "incorrect"]
        
        return (g_low in yes_vars and e_low in yes_vars) or \
               (g_low in no_vars and e_low in no_vars)
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not answer:
            return ""
        
        # Handle list format
        if "[" in answer and "]" in answer:
            match = re.search(r'\[([^\]]+)\]', answer)
            if match:
                content = re.sub(r'\s+', ' ', match.group(1)).strip()
                return f"[{content}]"
        
        # Normalize LaTeX
        answer = answer.replace(r'\dfrac', r'\frac')
        answer = re.sub(r'\\(left|right)([(\[\])])', r'\2', answer)
        
        # Remove spaces and commas
        answer = answer.replace(" ", "").replace(",", "")
        
        # Clean trailing zeros for decimals
        if "." in answer:
            answer = answer.rstrip("0").rstrip(".")
        
        return answer