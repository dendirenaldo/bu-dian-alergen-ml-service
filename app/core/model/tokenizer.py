from typing import List, Tuple

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class TextTokenizer:
    """Keras Tokenizer wrapper for text classification."""

    def __init__(self, vocab_size: int = 20000, max_len: int = 120):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")

    def fit(self, texts: List[str]) -> None:
        """Fit tokenizer on texts.

        Args:
            texts: List of raw text strings.
        """
        self.tokenizer.fit_on_texts(texts)

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert texts to integer sequences.

        Args:
            texts: List of raw text strings.

        Returns:
            List of integer sequences.
        """
        return self.tokenizer.texts_to_sequences(texts)

    def pad_sequences(
        self, sequences: List[List[int]], padding: str = "post", truncating: str = "post"
    ) -> np.ndarray:
        """Pad sequences to uniform length.

        Args:
            sequences: List of integer sequences.
            padding: Padding side ('post' or 'pre').
            truncating: Truncating side ('post' or 'pre').

        Returns:
            Padded numpy array of shape (n, max_len).
        """
        return pad_sequences(
            sequences, maxlen=self.max_len, padding=padding, truncating=truncating
        )

    def fit_and_pad(
        self, texts: List[str], padding: str = "post", truncating: str = "post"
    ) -> Tuple[np.ndarray, List[str]]:
        """Fit tokenizer and return padded sequences.

        Args:
            texts: List of raw text strings.
            padding: Padding side.
            truncating: Truncating side.

        Returns:
            Tuple of (padded sequences array, list of original texts).
        """
        self.fit(texts)
        sequences = self.texts_to_sequences(texts)
        padded = self.pad_sequences(sequences, padding=padding, truncating=truncating)
        return padded, texts

    @property
    def word_index(self):
        """Return the fitted word_index dictionary."""
        return self.tokenizer.word_index

    @property
    def num_words(self) -> int:
        """Return min of vocab_size and word_index length + 1."""
        return min(self.vocab_size, len(self.tokenizer.word_index) + 1)
