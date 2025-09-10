"""Prompt template management for different datasets and languages."""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Template for dataset-specific prompts."""
    template: str
    answer_prefix: str
    supports_multilingual: bool = False


class PromptTemplateManager:
    """
    Manages prompt templates for different datasets.
    
    Templates are designed for robust answer extraction with:
    - Clear answer format specifications to reduce parsing errors
    - Strict termination constraints to prevent trailing text after answers
    - Explicit format requirements for different answer types
    - Enhanced clarity for multiple choice and numerical answers
    """
    
    # Dataset-specific prompt templates with robust answer extraction constraints
    TEMPLATES = {
        "gsm8k": PromptTemplate(
            template="Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of \"Answer:\". Do not add anything other than the integer answer after \"Answer:\". Do not include any text after the final answer line.\n\nQuestion:\n{question}\n",
            answer_prefix="Answer",
            supports_multilingual=False
        ),
        "mgsm": PromptTemplate(
            template="Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of \"Answer:\". Do not add anything other than the numerical answer after \"Answer:\". Do not include any text after the final answer line.\n\nQuestion:\n{question}\n",
            answer_prefix="Answer",
            supports_multilingual=True
        ),
        "math": PromptTemplate(
            template="Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}. Put only the final numeric result inside \\boxed{{}}; do not add units or text inside the box.\n",
            answer_prefix="\\boxed",
            supports_multilingual=False
        ),
        "mmlu": PromptTemplate(
            template="Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering. Your final line must be exactly of the form 'Answer: X' where X ∈ {A,B,C,D} with no trailing text or punctuation.\n\nQuestion:\n{question}\n",
            answer_prefix="Answer",
            supports_multilingual=False
        ),
        "belebele": PromptTemplate(
            template="Answer the following multiple choice reading-comprehension question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Please fully understand the passage and give explanations step by step before answering. Your final line must be exactly of the form 'Answer: X' where X ∈ {A,B,C,D} with no trailing text or punctuation.\n\n{question}\n",
            answer_prefix="Answer",
            supports_multilingual=False
        ),
        "commonsenseqa": PromptTemplate(
            template="Answer the following multiple choice common-sense reasoning question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCDE. Think step by step and output the reasoning process before answering. Your final line must be exactly of the form 'Answer: X' where X ∈ {A,B,C,D,E} with no trailing text or punctuation.\n\n{question}",
            answer_prefix="Answer",
            supports_multilingual=False
        ),
        "hotpotqa": PromptTemplate(
            template="Answer the following multi-hop reasoning question. You need to combine information from multiple sources to find the answer. Think step by step and provide your reasoning process. The last line of your response should be of the following format: 'Answer: $ANSWER' (without quotes) where ANSWER is the final answer. Make the final answer a short noun phrase (≤ 5 words). If the question is yes/no, your final answer must be exactly 'Yes' or 'No'. Do not include any text after the final answer line.\n\nQuestion:\n{question}",
            answer_prefix="Answer",
            supports_multilingual=False
        ),
        "theoremqa": PromptTemplate(
            template="""Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Please read a math problem, and then think step by step to derive the answer. The answer is decided by Answer Type.
If the Answer type in [bool], the answer needs to be True or False.
Else if the Answer type in [integer, float] , The answer needs to be in numerical form.
Else if the Answer type in [list of integer, list of float] , the answer needs to be a list of number like [2, 3, 4].
Else if the Answer type in [option], the answer needs to be an option like (a), (b), (c), (d). Output one letter in parentheses, e.g. (a). Do not add text on that line.
You need to output the answer in your final sentence like 'Therefore, the answer is ...'. Do not include any text after the final answer sentence.

### Question: 
{question}

### Answer_type: {answer_type}

### Response:""",
            answer_prefix="Therefore, the answer is",
            supports_multilingual=False
        )
    }
    
    # Language-specific answer prefixes for multilingual datasets (MGSM)
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
    
    def get_prompt(self, 
                   dataset: str, 
                   question: str, 
                   answer_type: Optional[str] = None,
                   language: str = "en") -> str:
        """
        Get formatted prompt for dataset and language.
        """
        dataset = dataset.lower()
        
        if dataset not in self.TEMPLATES:
            # Fallback to generic template
            template = "{question}"
        else:
            template = self.TEMPLATES[dataset].template
        
        # Handle multilingual answer prefixes for MGSM dataset
        if (dataset == "mgsm" and language != "en" and 
            language in self.ANSWER_PREFIXES):
            localized_prefix = self.ANSWER_PREFIXES[language]
            template = template.replace("Answer:", f"{localized_prefix}:")
        
        # Format the template with question and answer type
        prompt = template.replace("{question}", question)
        
        # Replace answer type placeholder for TheoremQA
        if answer_type and "{answer_type}" in prompt:
            prompt = prompt.replace("{answer_type}", answer_type)
        
        return prompt
    
    def get_answer_prefix(self, dataset: str, language: str = "en") -> str:
        """Get answer prefix for dataset and language."""
        dataset = dataset.lower()
        
        if dataset == "mgsm" and language in self.ANSWER_PREFIXES:
            return self.ANSWER_PREFIXES[language]
        
        if dataset in self.TEMPLATES:
            return self.TEMPLATES[dataset].answer_prefix
        
        return "Answer"
    
    def is_multilingual_supported(self, dataset: str) -> bool:
        """Check if dataset supports multilingual prompts."""
        dataset = dataset.lower()
        return (dataset in self.TEMPLATES and 
                self.TEMPLATES[dataset].supports_multilingual)
    
    def list_supported_datasets(self) -> list:
        """Get list of supported datasets."""
        return list(self.TEMPLATES.keys())
    
    def list_supported_languages(self) -> list:
        """Get list of supported languages."""
        return list(self.ANSWER_PREFIXES.keys())
