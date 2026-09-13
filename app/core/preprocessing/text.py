import re
from typing import List

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


class TextPreprocessor:
    """Text preprocessing with Sastrawi stopwords and token filtering."""

    def __init__(self):
        # Lazy: stopword remover hanya dibuat saat preprocess() dipakai.
        # preprocess_v5 (BiLSTM final) tidak butuh Sastrawi sama sekali,
        # dan API Sastrawi berbeda antar versi (camelCase vs snake_case).
        self._stopword_remover = None
        self._stopwords = None

    def _ensure_stopwords(self):
        if self._stopwords is not None:
            return
        factory = StopWordRemoverFactory()
        if hasattr(factory, "get_stop_words"):
            self._stopwords = set(factory.get_stop_words())
        elif hasattr(factory, "get_stopwords"):
            self._stopwords = set(factory.get_stopwords())
        else:
            self._stopwords = set()
        for attr in ("createStopWordRemover", "create_stop_word_remover"):
            if hasattr(factory, attr):
                self._stopword_remover = getattr(factory, attr)()
                break

    def preprocess(self, text: str) -> List[str]:
        """Lowercase, remove non-alphanumeric, remove stopwords, filter short tokens.

        Args:
            text: Raw input text.

        Returns:
            List of filtered tokens.
        """
        text = text.lower()
        # FIX: ganti dengan spasi (bukan "") agar 'susu-bubuk' → 'susu bubuk'
        # (konsisten dengan simple_tokenize & training). "" menggabung kata.
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        self._ensure_stopwords()
        if self._stopword_remover is not None:
            text = self._stopword_remover.remove(text)
        tokens = text.split()
        tokens = [
            t for t in tokens if t not in self._stopwords and len(t) >= 2 and not t.isdigit()
        ]
        return tokens

    def preprocess_full(self, text: str) -> str:
        """Return preprocessed text as a single string.

        Args:
            text: Raw input text.

        Returns:
            Space-joined preprocessed tokens.
        """
        return " ".join(self.preprocess(text))

    def preprocess_v5(self, text: str) -> List[str]:
        """Tokenisasi parity model final V5 (WAJIB untuk BiLSTM V5).

        Model final (mask_zero, min_count=1, tanpa digit-fold) dilatih
        dengan ``simple_tokenize`` TANPA stopword removal. Memakai
        ``preprocess()`` (Sastrawi) di sini menyebabkan train-serve skew.
        """
        return self.simple_tokenize(text)

    def preprocess_v5_full(self, text: str) -> str:
        """Versi string dari :meth:`preprocess_v5`."""
        return " ".join(self.preprocess_v5(text))

    def simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, remove non-alphanumeric, split.

        Args:
            text: Raw input text.

        Returns:
            List of tokens.
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [tok for tok in text.split() if tok]

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """Filter tokens: remove stopwords, short, and digit-only tokens.

        Args:
            tokens: List of raw tokens.

        Returns:
            Filtered token list.
        """
        return [
            t
            for t in tokens
            if t not in self._stopwords and len(t) >= 2 and not t.isdigit()
        ]

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize whitespace and newlines.

        Args:
            text: Raw text.

        Returns:
            Normalized text.
        """
        text = text.replace("\r", " ")
        text = text.replace("\n", " ")
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def clean_phrase(text: str) -> str:
        """Lowercase, remove non-alphanumeric, normalize whitespace.

        Args:
            text: Raw text.

        Returns:
            Cleaned text.
        """
        text = TextPreprocessor.normalize_text(text.lower())
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
