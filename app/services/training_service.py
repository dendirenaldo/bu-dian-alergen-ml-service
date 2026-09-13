import logging
import os
import pickle
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from app.config import settings
from app.core.embedding.word2vec import Word2VecEmbedding
from app.core.model.architecture import build_model
from app.core.preprocessing.text import TextPreprocessor
from app.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class TrainingStatus:
    status: str = "idle"
    progress: float = 0.0
    epoch: int = 0
    total_epochs: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    message: str = ""
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None


class TrainingService:
    """Background training service with status tracking."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._status = TrainingStatus()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()

    @property
    def status(self) -> TrainingStatus:
        with self._lock:
            return self._status

    def start_training(
        self,
        data_path: str,
        text_col: str = "text",
        label_col: str = "label",
    ) -> None:
        """Start training in a background thread.

        Args:
            data_path: Path to training CSV.
            text_col: Name of text column.
            label_col: Name of label column.

        Raises:
            RuntimeError: If training is already in progress.
        """
        with self._lock:
            if self._status.status == "training":
                raise RuntimeError("Training already in progress")
            self._status = TrainingStatus(status="training", started_at=time.time())

        self._cancel_event.clear()
        self._thread = threading.Thread(
            target=self._train,
            args=(data_path, text_col, label_col),
            daemon=True,
        )
        self._thread.start()

    def cancel_training(self) -> None:
        """Request cancellation of the current training run."""
        with self._lock:
            if self._status.status == "training":
                self._cancel_event.set()
                self._status.message = "Cancellation requested"

    def _train(self, data_path: str, text_col: str, label_col: str) -> None:
        """Internal training routine."""
        try:
            logger.info(f"Loading training data from {data_path}")
            df = pd.read_csv(data_path, delimiter=getattr(settings, "CSV_DELIMITER", ";"))
            df = df[[text_col, label_col]].copy()
            df[text_col] = df[text_col].fillna("").astype(str)
            df[label_col] = df[label_col].fillna("").astype(str).str.strip().str.lower()
            df = df[df[text_col].str.strip() != ""].copy()
            df = df[df[label_col].isin(["safe", "unsafe"])].copy()
            df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
            if df.empty:
                raise ValueError("Dataset kosong setelah filtering.")
            if df[label_col].nunique() < 2:
                raise ValueError("Dataset hanya 1 kelas — training dibatalkan.")

            with self._lock:
                self._status.message = f"Loaded {len(df)} samples"

            label_encoder = LabelEncoder()
            df["label_id"] = label_encoder.fit_transform(df[label_col])

            X_train_text, X_test_text, y_train, y_test = train_test_split(
                df[text_col].tolist(),
                df["label_id"].values,
                test_size=settings.TEST_SIZE if hasattr(settings, "TEST_SIZE") else 0.2,
                random_state=42,
                stratify=df["label_id"].values,
            )

            preprocessor = TextPreprocessor()
            # FIX (CRITICAL train-serve skew + leakage): fit HANYA di train dan
            # di teks HASIL preprocess (sama seperti serving), bukan teks mentah
            # train+test. W2V tetap di token sederhana train-only.
            X_train_clean = [preprocessor.preprocess_full(t) for t in X_train_text]
            X_train_clean = [t for t in X_train_clean if t.strip()]
            if not X_train_clean:
                raise ValueError("Teks kosong setelah preprocessing.")
            tokenizer = Tokenizer(num_words=settings.VOCAB_SIZE, oov_token="<OOV>")
            tokenizer.fit_on_texts(X_train_clean)

            X_train_seq = tokenizer.texts_to_sequences(X_train_clean)
            X_test_clean = [preprocessor.preprocess_full(t) for t in X_test_text]
            X_test_seq = tokenizer.texts_to_sequences(X_test_clean)
            # OOV guard: cap indeks >= vocab ke 1.
            X_train_seq = [[i if i < settings.VOCAB_SIZE else 1 for i in s] for s in X_train_seq]
            X_test_seq = [[i if i < settings.VOCAB_SIZE else 1 for i in s] for s in X_test_seq]
            X_train_pad = pad_sequences(
                X_train_seq, maxlen=settings.MAX_LEN, padding="post", truncating="post"
            )
            X_test_pad = pad_sequences(
                X_test_seq, maxlen=settings.MAX_LEN, padding="post", truncating="post"
            )

            train_tokens = [preprocessor.simple_tokenize(t) for t in X_train_text]
            train_tokens = [preprocessor.filter_tokens(t) for t in train_tokens]

            w2v = Word2VecEmbedding(
                vector_size=settings.EMBED_DIM,
                window=settings.W2V_WINDOW,
                min_count=settings.MIN_WORD_COUNT,
                epochs=settings.W2V_EPOCHS,
            )
            w2v.train(train_tokens)

            num_words = min(settings.VOCAB_SIZE, len(tokenizer.word_index) + 1)
            embedding_matrix = w2v.build_embedding_matrix(tokenizer.word_index, num_words)

            with self._lock:
                self._status.message = "Building model"
                self._status.progress = 0.2

            model = build_model(
                num_words=num_words,
                embed_dim=settings.EMBED_DIM,
                embedding_matrix=embedding_matrix,
                use_bidirectional=True,
                lstm_units_1=128,
                lstm_units_2=64,
                dropout_1=0.3,
                dropout_2=0.3,
                lr=settings.LEARNING_RATE,
            )

            callbacks = [
                EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
                # FIX: checkpoint ke file temp agar training gagal tidak merusak
                # model produksi di MODEL_DIR.
                ModelCheckpoint(
                    os.path.join("/tmp", "bilstm_train_best.keras"),
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=0,
                ),
            ]

            with self._lock:
                self._status.message = "Training started"
                self._status.total_epochs = settings.EPOCHS
                self._status.progress = 0.3

            history = model.fit(
                X_train_pad,
                y_train,
                validation_split=0.15,
                epochs=settings.EPOCHS,
                batch_size=settings.BATCH_SIZE,
                callbacks=callbacks,
                verbose=1,
            )

            if self._cancel_event.is_set():
                with self._lock:
                    self._status.status = "cancelled"
                    self._status.message = "Training cancelled"
                    self._status.finished_at = time.time()
                logger.info("Training cancelled by user")
                return

            with self._lock:
                self._status.progress = 0.8
                self._status.message = "Saving models"

            os.makedirs(settings.MODEL_DIR, exist_ok=True)

            model.save(os.path.join(settings.MODEL_DIR, "bilstm_model.keras"))
            w2v.save(os.path.join(settings.MODEL_DIR, "word2vec.model"))

            with open(os.path.join(settings.MODEL_DIR, "tokenizer.pkl"), "wb") as f:
                pickle.dump(tokenizer, f)
            with open(os.path.join(settings.MODEL_DIR, "label_encoder.pkl"), "wb") as f:
                pickle.dump(label_encoder, f)

            self.registry.load_models(settings.MODEL_DIR)

            val_acc = max(history.history.get("val_accuracy", [0]))
            val_loss = min(history.history.get("val_loss", [float("inf")]))

            with self._lock:
                self._status.status = "completed"
                self._status.progress = 1.0
                self._status.accuracy = float(val_acc)
                self._status.val_loss = float(val_loss)
                self._status.message = "Training completed"
                self._status.finished_at = time.time()

            logger.info("Training completed successfully")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            with self._lock:
                self._status.status = "failed"
                self._status.error = str(e)
                self._status.message = "Training failed"
                self._status.finished_at = time.time()
