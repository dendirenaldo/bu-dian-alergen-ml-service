import logging
import os
import pickle
import threading
from typing import Optional

from gensim.models import Word2Vec
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Singleton registry for loaded ML models in memory."""

    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._loaded = False
                cls._instance._bi_lstm_model = None
                cls._instance._word2vec_model = None
                cls._instance._tokenizer = None
                cls._instance._label_encoder = None
            return cls._instance

    def load_models(self, model_dir: str) -> None:
        """Load all models from disk.

        Args:
            model_dir: Directory containing saved model files.
        """
        with self._lock:
            self._load_models_internal(model_dir)

    def _load_models_internal(self, model_dir: str) -> None:
        try:
            model_path = os.path.join(model_dir, "bilstm_model.keras")
            w2v_path = os.path.join(model_dir, "word2vec.model")
            tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
            le_path = os.path.join(model_dir, "label_encoder.pkl")

            if os.path.exists(model_path):
                self._bi_lstm_model = load_model(model_path)
                logger.info("BiLSTM model loaded")

            if os.path.exists(w2v_path):
                self._word2vec_model = Word2Vec.load(w2v_path)
                logger.info("Word2Vec model loaded")

            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, "rb") as f:
                    self._tokenizer = pickle.load(f)
                logger.info("Tokenizer loaded")

            if os.path.exists(le_path):
                with open(le_path, "rb") as f:
                    self._label_encoder = pickle.load(f)
                logger.info("Label encoder loaded")

            # _loaded True hanya jika artefak kritis tersedia (hindari None.predict).
            has_core = self._bi_lstm_model is not None and self._tokenizer is not None
            self._loaded = bool(has_core)
            if has_core:
                logger.info(f"All models loaded from {model_dir}")
            else:
                logger.warning(
                    f"Model tidak lengkap di {model_dir} "
                    f"(bilstm={self._bi_lstm_model is not None}, "
                    f"tokenizer={self._tokenizer is not None}). "
                    "Endpoint /detection akan 503 sampai model lengkap."
                )
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self._loaded = False

    def save_models(self, model_dir: str) -> None:
        """Save all models to disk.

        Args:
            model_dir: Directory to save model files.
        """
        with self._lock:
            self._save_models_internal(model_dir)

    def _save_models_internal(self, model_dir: str) -> None:
        os.makedirs(model_dir, exist_ok=True)

        if self._bi_lstm_model:
            self._bi_lstm_model.save(os.path.join(model_dir, "bilstm_model.keras"))
        if self._word2vec_model:
            self._word2vec_model.save(os.path.join(model_dir, "word2vec.model"))
        if self._tokenizer:
            with open(os.path.join(model_dir, "tokenizer.pkl"), "wb") as f:
                pickle.dump(self._tokenizer, f)
        if self._label_encoder:
            with open(os.path.join(model_dir, "label_encoder.pkl"), "wb") as f:
                pickle.dump(self._label_encoder, f)

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._loaded and self._bi_lstm_model is not None and self._tokenizer is not None

    def snapshot(self):
        """Ambil (model, tokenizer) secara atomik untuk hindari TOCTOU reload."""
        with self._lock:
            return self._bi_lstm_model, self._tokenizer, self._label_encoder, self._loaded

    @property
    def bi_lstm_model(self):
        return self._bi_lstm_model

    @property
    def word2vec_model(self):
        return self._word2vec_model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def label_encoder(self):
        return self._label_encoder
