"""Prompt template management for different datasets and languages."""

from typing import Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Template for dataset-specific prompts."""
    template: str
    answer_prefix: str
    supports_multilingual: bool = False


class PromptTemplateManager:
    """Manages prompt templates for different datasets."""
    
    TEMPLATES = {
        "gsm8k": PromptTemplate(
            template="Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of \"Answer:\". Do not add anything other than the integer answer after \"Answer:\".\n\nQuestion:\n{question}\n",
            answer_prefix="Answer",
            supports_multilingual=False
        ),
        "mgsm": PromptTemplate(
            template="Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of \"Answer:\". Do not add anything other than the integer answer after \"Answer:\".\n\nQuestion:\n{question}\n",
            answer_prefix="Answer",
            supports_multilingual=True
        ),
        "math": PromptTemplate(
            template="Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n",
            answer_prefix="\\boxed",
            supports_multilingual=False
        ),
        "mmlu": PromptTemplate(
            template="Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\nQuestion:\n{question}\n",
            answer_prefix="Answer",
            supports_multilingual=False
        ),
        "belebele": PromptTemplate(
            template="Answer the following multiple choice reading-comprehension question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Please fully understand the passage and give explanations step by step before answering.\n\n{question}\n",
            answer_prefix="Answer",
            supports_multilingual=False
        ),
        "commonsenseqa": PromptTemplate(
            template="Answer the following multiple choice common-sense reasoning question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCDE. Think step by step and output the reasoning process before answering.\n\n{question}",
            answer_prefix="Answer",
            supports_multilingual=False
        ),
        "hotpotqa": PromptTemplate(
            template="Answer the following multi-hop reasoning question. Think step by step and provide your reasoning process. The last line of your response should be of the following format: 'Answer: $ANSWER' (without quotes) where ANSWER is the final answer.\n\nQuestion:\n{question}",
            answer_prefix="Answer",
            supports_multilingual=False
        ),
        "theoremqa": PromptTemplate(
            template="""Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Please read a math problem, and then think step by step to derive the answer. The answer is decided by Answer Type.
If the Answer type in [bool], the answer needs to be True or False.
Else if the Answer type in [integer, float], The answer needs to be in numerical form.
Else if the Answer type in [list of integer, list of float], the answer needs to be a list of number like [2, 3, 4].
Else if the Answer type in [option], the answer needs to be an option like (a), (b), (c), (d).
You need to output the answer in your final sentence like 'Therefore, the answer is ...'.

### Question: 
{question}

### Answer_type: {answer_type}

### Response:""",
            answer_prefix="Therefore, the answer is",
            supports_multilingual=False
        )
    }
    
    # Language-specific answer prefixes for multilingual datasets
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
        """Get formatted prompt for dataset and language."""
        dataset = dataset.lower()
        
        if dataset not in self.TEMPLATES:
            # Fallback to generic template
            template = "{question}"
        else:
            template = self.TEMPLATES[dataset].template
        
        # Handle multilingual answer prefixes for MGSM
        if dataset == "mgsm" and language != "en" and language in self.ANSWER_PREFIXES:
            localized_prefix = self.ANSWER_PREFIXES[language]
            template = template.replace("Answer:", f"{localized_prefix}:")
        
        # Format template
        prompt = template.replace("{question}", question)
        
        # Replace answer type for TheoremQA
        if answer_type and "{answer_type}" in prompt:
            prompt = prompt.replace("{answer_type}", answer_type)
        
        return prompt