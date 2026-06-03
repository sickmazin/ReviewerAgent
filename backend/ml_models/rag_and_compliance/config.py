import os
import re

FALLBACK_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "microsoft/deberta-v3-small",
]

GUIDELINES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", ".guidelines"))

MODEL_NAME = "gemma3:27b"

# Shared Regex patterns
URL_RE   = re.compile(r'https?://\S+|www\.\S+', re.I)
EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', re.I)
PHONE_RE = re.compile(r'(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}')
CAPS_RE  = re.compile(r'[A-ZÀÈÉÌÒÙ]{5,}')


# PROMPT

prompt_file_path = os.path.join(os.path.dirname(__file__), "../..", ".prompts", "grammar_evaluation.txt")
with open(prompt_file_path, "r", encoding="utf-8") as f:
    GRAMMAR_PROMPT = f.read()


prompt_file_path = os.path.join(os.path.dirname(__file__), "../..", ".prompts", "guidelines.txt")
with open(prompt_file_path, "r", encoding="utf-8") as f:
    GUIDELINES_PROMPT = f.read()