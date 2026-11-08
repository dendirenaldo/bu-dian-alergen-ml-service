import re
from typing import List


class TextPreprocessor:
    def preprocess_v5(self, text: str) -> List[str]:
        return self.simple_tokenize(text)

    def preprocess_v5_full(self, text: str) -> str:
        return " ".join(self.preprocess_v5(text))

    def simple_tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [tok for tok in text.split() if tok]

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.replace("\r", " ")
        text = text.replace("\n", " ")
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def clean_phrase(text: str) -> str:
        text = TextPreprocessor.normalize_text(text.lower())
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
