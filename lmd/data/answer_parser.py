"""Answer parsing and validation utilities."""

import re
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..utils.types import SampleRecord

# Unified numerical regex pattern (supports .5 / scientific notation)
NUMBER_RE = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


@dataclass
class ParseResult:
    """Result of answer parsing."""
    extracted_answer: str
    is_correct: bool
    confidence: Optional[float] = None


class AnswerParser:
    """Robust answer extraction and validation for multiple datasets."""
    
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
        """Initialize answer parser for specific dataset."""
        self.dataset = dataset.lower()
    
    def parse_and_label(self, generated_text: str, sample: SampleRecord) -> Dict[str, Any]:
        """Parse generated text and validate correctness."""
        extracted = self._extract_answer(generated_text, sample)
        is_correct = self._validate_answer(extracted, sample.answer_gt, sample)
        
        return {
            "extracted_answer": extracted,
            "correct": is_correct
        }
    
    def _extract_answer(self, text: str, sample: SampleRecord) -> str:
        """Extract answer based on dataset-specific patterns."""
        if self.dataset in ["mgsm", "gsm8k"]:
            return self._extract_numerical_answer(text, sample.language)
        elif self.dataset == "math":
            return self._extract_math_answer(text)
        elif self.dataset in ["commonsenseqa", "mmlu", "belebele"]:
            return self._extract_multiple_choice_answer(text, self.dataset)
        elif self.dataset == "hotpotqa":
            return self._extract_hotpotqa_answer(text)
        elif self.dataset == "theoremqa":
            return self._extract_theoremqa_answer(text, sample.answer_type)
        else:
            return self._extract_generic_answer(text)
    
    def _extract_numerical_answer(self, text: str, language: str) -> str:
        """Extract numerical answer with language support."""
        localized_prefix = self.ANSWER_PREFIXES.get(language, "Answer")

        # Extract from "Answer:" / "answer is" / "Therefore, the answer is"
        # Use LAST match to avoid section headers
        patterns = [
            rf'(?:^|\n)\s*{re.escape(localized_prefix)}:\s*(.+?)(?:(?<!\d)\.(?!\d)|$)',
            r'(?:^|\n)\s*Answer:\s*(.+?)(?:(?<!\d)\.(?!\d)|$)',
            r'(?:Therefore,?\s*)?(?:the\s+)?answer\s+is\s*(.+?)(?:(?<!\d)\.(?!\d)|$)',
        ]

        for pat in patterns:
            matches = list(re.finditer(pat, text, re.IGNORECASE | re.DOTALL))
            if matches:
                # Take the LAST match to avoid section headers
                m = matches[-1]
                cand = m.group(1).strip().replace(",", "")
                nums = re.findall(NUMBER_RE, cand)
                if nums:
                    return nums[-1].rstrip(".")
                # No numbers found, try next pattern
                continue

        # Fallback: last number in entire text (supports .5 / scientific notation)
        nums = re.findall(NUMBER_RE, text.replace(",", ""))
        return nums[-1].rstrip(".") if nums else ""
    
    def _extract_math_answer(self, text: str) -> str:
        """Extract answer from \\boxed{...} pattern."""
        pattern = re.compile(r'\\boxed\{')
        matches = list(pattern.finditer(text))
        
        if not matches:
            return ""
        
        # Take the last boxed expression
        match = matches[-1]
        start_pos = match.end()
        brace_count = 1
        i = start_pos
        
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
        
        if brace_count == 0:
            ans = text[start_pos:i-1].strip()
            # Remove outer braces (handle \boxed{{2}} case)
            while ans.startswith("{") and ans.endswith("}"):
                ans = ans[1:-1].strip()
            # Remove \text{...} content
            ans = re.sub(r'\\text\{[^}]*\}', '', ans)
            return ans
        
        return ""
    
    def _extract_multiple_choice_answer(self, text: str, dataset: str) -> str:
        """Extract multiple choice answer."""
        n_choices = 5 if dataset == "commonsenseqa" else 4
        choices_str = "ABCDE"[:n_choices]
        
        patterns = [
            rf"Answer:\s*([{choices_str}])",
            rf"answer is\s*\(?([{choices_str}])\)?",
            rf"Therefore,?\s*(?:the answer is)?\s*\(?([{choices_str}])\)?",
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                # Take the LAST match to avoid intermediate answers
                return matches[-1].group(1).upper()
        
        return ""
    
    def _extract_hotpotqa_answer(self, text: str) -> str:
        """Extract HotpotQA answer with flexible matching."""
        patterns = [
            r"Answer:\s*(.+?)(?:\.|$)",
            r"answer is\s*(.+?)(?:\.|$)",
            r"Therefore,?\s*(?:the answer is)?\s*(.+?)(?:\.|$)",
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                # Take the LAST match to avoid intermediate answers
                answer = matches[-1].group(1).strip()
                return re.sub(r"\s*\.$", "", answer)
        
        return ""
    
    def _extract_theoremqa_answer(self, text: str, answer_type: Optional[str]) -> str:
        """
        More robust TheoremQA extraction logic:
        1) Prioritize \boxed{...}
        2) Then match "answer is ...", termination condition doesn't treat decimal points as sentence endings
        3) Extract numerical types using NUMBER_RE (supports .5 / scientific notation)
        """
        # 1) Prioritize \boxed{...} (cross-line)
        boxed = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text, re.DOTALL)
        candidate = None
        if boxed:
            candidate = boxed.group(1).strip()
        else:
            # 2) "answer is ..." (cross-line, termination doesn't treat decimal points as sentence endings)
            m = re.search(
                r'(?:Therefore,?\s*)?(?:the\s+)?answer\s+is\s*(.+?)(?:(?<!\d)\.(?!\d)|$)',
                text, re.IGNORECASE | re.DOTALL
            )
            if m:
                candidate = m.group(1).strip()
            else:
                # 3) Fallback: "Answer:" line (also doesn't treat decimal points as sentence endings)
                m2 = re.search(
                    r'(?:^|\n)\s*Answer\s*:\s*(.+?)(?:(?<!\d)\.(?!\d)|$)',
                    text, re.IGNORECASE | re.DOTALL
                )
                if m2:
                    candidate = m2.group(1).strip()

        if not candidate:
            # Fallback: try "Therefore ... is <number>" pattern
            m = re.search(r'Therefore[^.\n]{0,120}?is\s*(' + NUMBER_RE + r')', text, re.IGNORECASE | re.DOTALL)
            if m:
                return m.group(1).rstrip(".")
            
            # Fallback: look for last number in final 1-2 sentences
            parts = re.split(r'(?<!\d)\.(?!\d)', text)
            tail = ".".join(parts[-2:]) if len(parts) >= 2 else text
            nums = re.findall(NUMBER_RE, tail.replace(",", ""))
            if nums:
                return nums[-1].rstrip(".")
            return ""

        # Clean up LaTeX and miscellaneous items
        cand = candidate
        cand = cand.replace("$", "")
        # \frac{a}{b} -> try to convert to numerical value
        frac = re.search(r'\\frac\{([^}]+)\}\{([^}]+)\}', cand)
        if frac:
            try:
                num = float(frac.group(1))
                den = float(frac.group(2))
                cand = cand.replace(frac.group(0), str(num / den))
            except Exception:
                pass
        # First remove other LaTeX commands and extra parentheses/braces
        cand = re.sub(r"\\[a-zA-Z]+", "", cand)     # Remove command names
        cand = cand.replace("\\(", "").replace("\\)", "")
        cand = cand.replace("{", "").replace("}", "")
        
        # \pi -> 3.14159; also compatible with 'pi', but need to handle number+pi cases
        # First handle number+pi cases, e.g. 0.5\pi -> 0.5*3.14159
        cand = re.sub(r'(\d+(?:\.\d+)?)\\pi', r'\1*3.14159', cand)
        cand = re.sub(r'(\d+(?:\.\d+)?)pi', r'\1*3.14159', cand)
        # Then handle standalone \pi
        cand = cand.replace(r"\pi", "3.14159").replace("pi", "3.14159")
        
        # Try to calculate simple mathematical expressions (e.g. 0.5*3.14159)
        try:
            # Only calculate simple multiplication and division
            if '*' in cand and all(c in '0123456789.*+-' for c in cand.replace(' ', '')):
                result = eval(cand)
                if isinstance(result, (int, float)):
                    cand = str(result)
        except:
            pass
        
        cand = re.sub(r"\s+", " ", cand).strip()

        # Post-process based on type
        if answer_type == "bool":
            low = cand.lower()
            if any(x in low for x in ["true", "yes", "correct", "right"]):  return "True"
            if any(x in low for x in ["false", "no", "incorrect", "wrong"]): return "False"
            return ""

        if answer_type in ["integer", "float"]:
            nums = re.findall(NUMBER_RE, cand.replace(",", ""))
            return nums[-1].rstrip(".") if nums else ""

        if answer_type in ["list of integer", "list of float"]:
            mlist = re.search(r"\[([^\]]+)\]", cand, re.DOTALL)
            return mlist.group(1).strip() if mlist else ""

        if answer_type == "option":
            mopt = re.search(r"\(([a-dA-D])\)", cand)
            return mopt.group(1).upper() if mopt else ""

        # Fallback: return the cleaned string as-is
        return cand
    
    def _extract_generic_answer(self, text: str) -> str:
        """Generic answer extraction fallback."""
        # Look for common answer patterns
        patterns = [
            r"Answer:\s*(.+?)(?:\.|$)",
            r"answer is\s*(.+?)(?:\.|$)",
            r"Therefore,?\s*(?:the answer is)?\s*(.+?)(?:\.|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _validate_answer(self, extracted: str, ground_truth: str, sample: SampleRecord) -> bool:
        """Validate extracted answer against ground truth."""
        if not extracted:
            return False
        
        # First perform unified normalization
        extracted_norm = self._normalize_answer(extracted)
        gt_norm = self._normalize_answer(ground_truth)

        # TheoremQA numerical questions also use numerical validation
        if self.dataset == "theoremqa" and (sample.answer_type in ["integer", "float"]):
            return self._validate_numerical_answer(extracted_norm, gt_norm)

        # Dataset-specific validation
        if self.dataset in ["mgsm", "gsm8k"]:
            return self._validate_numerical_answer(extracted_norm, gt_norm)
        elif self.dataset == "math":
            # For math dataset, try numerical validation first, fallback to string comparison
            if self._validate_numerical_answer(extracted_norm, gt_norm):
                return True
            return extracted_norm.lower() == gt_norm.lower()
        elif self.dataset == "hotpotqa":
            return self._validate_hotpotqa_answer(extracted_norm, gt_norm)
        else:
            return extracted_norm.lower() == gt_norm.lower()
    
    def _validate_numerical_answer(self, extracted: str, ground_truth: str) -> bool:
        """Validate numerical answers."""
        try:
            extracted_num = float(extracted) if extracted else None
            ground_truth_num = float(ground_truth) if ground_truth else None
            
            if extracted_num is not None and ground_truth_num is not None:
                return abs(extracted_num - ground_truth_num) < 1e-6
        except ValueError:
            pass
        
        return False
    
    def _validate_hotpotqa_answer(self, extracted: str, ground_truth: str) -> bool:
        """Validate HotpotQA answers with flexible matching."""
        extracted_lower = extracted.lower().strip()
        ground_truth_lower = ground_truth.lower().strip()
        
        # Exact match
        if extracted_lower == ground_truth_lower:
            return True
        
        # Substring matches
        if ground_truth_lower in extracted_lower or extracted_lower in ground_truth_lower:
            return True
        
        # Yes/No variations
        yes_variations = ["yes", "true", "correct", "right"]
        no_variations = ["no", "false", "incorrect", "wrong"]
        
        if (ground_truth_lower in yes_variations and extracted_lower in yes_variations or
            ground_truth_lower in no_variations and extracted_lower in no_variations):
            return True
        
        return False
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not answer:
            return ""
        
        import re
        
        # For list types, uniformly handle square brackets and commas
        if "[" in answer and "]" in answer:
            # Extract content within square brackets, remove spaces, keep comma separation
            list_match = re.search(r'\[([^\]]+)\]', answer)
            if list_match:
                content = list_match.group(1).strip()
                # Remove extra spaces but keep comma separation
                content = re.sub(r'\s+', ' ', content).strip()
                return f"[{content}]"
        elif "," in answer:
            # If contains commas, might be list format, uniformly add square brackets
            content = re.sub(r'\s+', ' ', answer).strip()
            return f"[{content}]"
        
        # Normalize LaTeX math expressions
        answer = self._normalize_latex_math(answer)
        
        # For regular answers, remove spaces and commas
        answer = answer.replace(" ", "").replace(",", "")
        
        # Strip trailing zeros and decimal point for numbers
        if "." in answer:
            answer = answer.rstrip("0").rstrip(".")
        
        return answer
    
    def _normalize_latex_math(self, answer: str) -> str:
        """Normalize LaTeX math expressions for comparison."""
        if not answer:
            return answer
        
        # Convert \dfrac to \frac (they are mathematically equivalent)
        answer = answer.replace(r'\dfrac', r'\frac')
        
        # Normalize other common LaTeX variations
        answer = answer.replace(r'\left(', '(')
        answer = answer.replace(r'\right)', ')')
        answer = answer.replace(r'\left[', '[')
        answer = answer.replace(r'\right]', ']')
        
        # Remove extra spaces around operators
        answer = re.sub(r'\s*([+\-*/=<>])\s*', r'\1', answer)
        
        return answer
