import json
import logging
import os
import pickle
import threading
from typing import Any, Optional

from gensim.models import Word2Vec
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)


class _JsonTokenizer:
    """Adapter tokenizer dari JSON export training (tanpa pickle).

    Pickle lintas repo rapuh (path kelas sama, isi beda). JSON berisi
    {vocab_size, max_len, word_index} direkonstruksi ke Keras Tokenizer.
    """

    def __init__(self, vocab_size: int, max_len: int, word_index: dict):
        from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer

        self.vocab_size = vocab_size
        self.max_len = max_len
        tok = KerasTokenizer(num_words=vocab_size, oov_token="<OOV>")
        tok.word_index = dict(word_index)
        # Pastikan <OOV> ada (penting bila JSON dibuat manual/vocab kecil).
        tok.word_index.setdefault("<OOV>", 1)
        tok.index_word = {int(v): k for k, v in tok.word_index.items()}
        self._tokenizer = tok

    def texts_to_sequences(self, texts):
        seqs = self._tokenizer.texts_to_sequences(texts)
        return [[i if i < self.vocab_size else 1 for i in s] for s in seqs]

    @property
    def word_index(self):
        return self._tokenizer.word_index


def _load_tokenizer_json(path: str):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    return _JsonTokenizer(
        vocab_size=int(cfg["vocab_size"]),
        max_len=int(cfg["max_len"]),
        word_index=dict(cfg["word_index"]),
    )


class ModelRegistry:
    """Singleton registry dual-model (BiLSTM + BERT) di memori."""

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
                cls._instance._thresholds: dict = {}
                cls._instance._metadata: dict = {}
                # BERT (lazy, opsional)
                cls._instance._bert_model = None
                cls._instance._bert_tokenizer = None
                cls._instance._bert_threshold: float = 0.5
                cls._instance._bert_model_name: str = ""
            return cls._instance

    # -- kompatibilitas: reset untuk testing --
    def reset(self) -> None:
        """Reset seluruh state (dipakai testing)."""
        with self._lock:
            self._loaded = False
            self._bi_lstm_model = None
            self._word2vec_model = None
            self._tokenizer = None
            self._label_encoder = None
            self._thresholds = {}
            self._metadata = {}
            self._bert_model = None
            self._bert_tokenizer = None
            self._bert_threshold = 0.5
            self._bert_model_name = ""

    def load_models(self, model_dir: str) -> None:
        """Load all models from disk.

        Args:
            model_dir: Directory containing saved model files.
        """
        with self._lock:
            self._load_models_internal(model_dir)

    def _load_models_internal(self, model_dir: str) -> None:
        # FIX: bersihkan state lama dulu agar gagal-load tidak menyisakan
        # model/tokenizer setengah (state divergen).
        self._bi_lstm_model = None
        self._word2vec_model = None
        self._tokenizer = None
        self._label_encoder = None
        self._thresholds = {}
        self._metadata = {}
        try:
            model_path = os.path.join(model_dir, "bilstm_model.keras")
            w2v_path = os.path.join(model_dir, "word2vec.model")
            tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
            le_path = os.path.join(model_dir, "label_encoder.pkl")
            thr_path = os.path.join(model_dir, "thresholds.json")
            meta_path = os.path.join(model_dir, "metadata.json")

            if os.path.exists(model_path):
                self._bi_lstm_model = load_model(model_path)
                logger.info("BiLSTM model loaded")

            if os.path.exists(w2v_path):
                self._word2vec_model = Word2Vec.load(w2v_path)
                logger.info("Word2Vec model loaded")

            if os.path.exists(tokenizer_path):
                loaded_tok = None
                # Jalur utama: JSON (robust lintas repo). Pickle hanya fallback.
                json_path = os.path.join(model_dir, "tokenizer_bilstm.json")
                if os.path.exists(json_path):
                    try:
                        loaded_tok = _load_tokenizer_json(json_path)
                        logger.info("Tokenizer loaded (JSON)")
                    except Exception as e:
                        logger.warning("Gagal load tokenizer JSON: %s", e)
                if loaded_tok is None:
                    try:
                        with open(tokenizer_path, "rb") as f:
                            cand = pickle.load(f)
                        # Validasi: objek harus bisa texts_to_sequences.
                        cand.texts_to_sequences(["uji coba"])
                        loaded_tok = cand
                        logger.info("Tokenizer loaded (pickle)")
                    except Exception as e:
                        logger.warning("Gagal load tokenizer pickle: %s", e)
                self._tokenizer = loaded_tok

            if os.path.exists(le_path):
                with open(le_path, "rb") as f:
                    self._label_encoder = pickle.load(f)
                logger.info("Label encoder loaded")

            if os.path.exists(thr_path):
                try:
                    with open(thr_path, encoding="utf-8") as f:
                        self._thresholds = dict(json.load(f))
                    logger.info("Thresholds loaded: %s", self._thresholds)
                except Exception as e:
                    logger.warning("Gagal load thresholds.json: %s", e)

            if os.path.exists(meta_path):
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        self._metadata = dict(json.load(f))
                except Exception as e:
                    logger.warning("Gagal load metadata.json: %s", e)

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
            # Coba load BERT juga (opsional, tidak menggagalkan BiLSTM).
            try:
                from app.config import settings as _settings
                self._load_bert_internal(
                    getattr(_settings, "BERT_MODEL_DIR", os.path.join(model_dir, "bert")),
                    getattr(_settings, "BERT_MODEL_NAME", ""),
                )
            except Exception as e:
                logger.info("BERT tidak dimuat (opsional): %s", e)
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self._bi_lstm_model = None
            self._tokenizer = None
            self._loaded = False

    def _load_bert_internal(self, bert_dir: str, fallback_name: str = "") -> bool:
        """Load model BERT dari direktori HF (lazy transformers)."""
        if not bert_dir or not os.path.isdir(bert_dir):
            return False
        if not (os.path.exists(os.path.join(bert_dir, "config.json"))):
            return False
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            logger.warning("transformers/torch belum terinstal, BERT dilewati: %s", e)
            return False
        try:
            self._bert_tokenizer = AutoTokenizer.from_pretrained(bert_dir, use_fast=True)
            self._bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
            self._bert_model.eval()
            thr_file = os.path.join(bert_dir, "threshold.json")
            if os.path.exists(thr_file):
                with open(thr_file, encoding="utf-8") as f:
                    self._bert_threshold = float(json.load(f).get("threshold", 0.5))
            meta_file = os.path.join(bert_dir, "metrics.json")
            name = fallback_name
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, encoding="utf-8") as f:
                        name = json.load(f).get("model_name", name)
                except Exception:
                    pass
            self._bert_model_name = name or bert_dir
            logger.info("BERT model loaded: %s (thr=%.3f)",
                        self._bert_model_name, self._bert_threshold)
            return True
        except Exception as e:
            logger.warning("Gagal load BERT dari %s: %s", bert_dir, e)
            self._bert_model = None
            self._bert_tokenizer = None
            return False

    def load_bert(self, bert_dir: str) -> bool:
        """API publik untuk (re)load BERT."""
        with self._lock:
            return self._load_bert_internal(bert_dir)

    def is_ready(self, model_name: str = "bilstm") -> bool:
        """Cek kesiapan per model: 'bilstm' | 'bert' | 'ensemble'."""
        with self._lock:
            if model_name == "bert":
                return self._bert_model is not None and self._bert_tokenizer is not None
            if model_name == "ensemble":
                return (self._bi_lstm_model is not None and self._tokenizer is not None
                        and self._bert_model is not None)
            return self._loaded and self._bi_lstm_model is not None and self._tokenizer is not None

    def bert_snapshot(self):
        """Ambil (bert_model, bert_tokenizer, threshold, nama) atomik."""
        with self._lock:
            return (self._bert_model, self._bert_tokenizer,
                    self._bert_threshold, self._bert_model_name)

    def get_threshold(self, model_name: str = "bilstm", default: float = 0.5) -> float:
        """Threshold per model dari thresholds.json (fallback default)."""
        with self._lock:
            if model_name == "bert":
                return float(self._bert_threshold or default)
            try:
                return float(self._thresholds.get(model_name, default))
            except Exception:
                return default

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

    @property
    def bert_model(self):
        return self._bert_model

    @property
    def bert_tokenizer(self):
        return self._bert_tokenizer

    @property
    def metadata(self) -> dict:
        with self._lock:
            return dict(self._metadata)
