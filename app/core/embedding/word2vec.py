import logging
from typing import Dict, List, Optional

import numpy as np
from gensim.models import Word2Vec

logger = logging.getLogger(__name__)


class Word2VecEmbedding:
    """Word2Vec training and embedding matrix construction."""

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 3,
        epochs: int = 30,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model: Optional[Word2Vec] = None

    def train(self, sentences: List[List[str]]) -> None:
        """Train Word2Vec model on tokenized sentences.

        Args:
            sentences: List of tokenized sentences.
        """
        self.model = Word2Vec(
            sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            sg=1,
            epochs=self.epochs,
            seed=42,
        )
        logger.info(
            f"Word2Vec trained. Vocabulary size: {len(self.model.wv)}"
        )

    def build_embedding_matrix(
        self, word_index: Dict[str, int], vocab_size: int
    ) -> np.ndarray:
        """Build embedding matrix from trained Word2Vec and Keras word_index.

        Words not found in Word2Vec get random initialization.

        Args:
            word_index: Keras tokenizer word_index mapping.
            vocab_size: Maximum vocabulary size.

        Returns:
            Embedding matrix of shape (vocab_size, vector_size).
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        embedding_matrix = np.random.normal(
            scale=0.6, size=(vocab_size, self.vector_size)
        ).astype(np.float32)
        embedding_matrix[0] = np.zeros((self.vector_size,), dtype=np.float32)

        found = 0
        for word, idx in word_index.items():
            if idx >= vocab_size:
                continue
            if word in self.model.wv:
                embedding_matrix[idx] = self.model.wv[word]
                found += 1

        logger.info(
            f"Embedding matrix: {found}/{len(word_index)} words found in Word2Vec"
        )
        return embedding_matrix

    def save(self, path: str) -> None:
        """Save Word2Vec model to disk.

        Args:
            path: File path to save the model.
        """
        if self.model:
            self.model.save(path)

    def load(self, path: str) -> None:
        """Load Word2Vec model from disk.

        Args:
            path: File path to load the model from.
        """
        self.model = Word2Vec.load(path)
