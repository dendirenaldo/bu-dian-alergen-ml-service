import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from app.core.model.architecture import build_model

logger = logging.getLogger(__name__)

# Hyperparameter search space
BATCH_SIZE_OPTIONS = [16, 32, 64]
LSTM_UNITS_OPTIONS = [[32, 64], [16, 32], [64, 64]]
DROPOUT_OPTIONS = [[0.1, 0.1], [0.3, 0.3], [0.0, 0.0]]
TUNING_EPOCHS_OPTIONS = [50, 75, 100]
LR_OPTIONS = [1e-4, 1e-3]


class ModelTuner:
    """Hyperparameter tuning via random search for BiLSTM/LSTM models."""

    def __init__(
        self,
        num_words: int,
        embed_dim: int,
        embedding_matrix,
        num_trials: int = 10,
        val_split: float = 0.15,
        recurrent_dropout: float = 0.0,
        embed_trainable: bool = False,
        gradient_clip_norm: float = 1.0,
        class_weight: Optional[Dict[int, float]] = None,
        early_stopping_patience: int = 5,
        lr_reduce_patience: int = 3,
        min_lr: float = 1e-6,
    ):
        self.num_words = num_words
        self.embed_dim = embed_dim
        self.embedding_matrix = embedding_matrix
        self.num_trials = num_trials
        self.val_split = val_split
        self.recurrent_dropout = recurrent_dropout
        self.embed_trainable = embed_trainable
        self.gradient_clip_norm = gradient_clip_norm
        self.class_weight = class_weight
        self.early_stopping_patience = early_stopping_patience
        self.lr_reduce_patience = lr_reduce_patience
        self.min_lr = min_lr

    def tune_and_train(
        self,
        use_bidirectional: bool,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        max_epochs: int = 150,
        seed: int = 42,
    ) -> Tuple[tf.keras.Model, dict]:
        """Run random search tuning then train final model.

        Args:
            use_bidirectional: If True, use Bidirectional LSTM.
            model_name: Name for logging/checkpointing.
            X_train: Training padded sequences.
            y_train: Training labels.
            max_epochs: Maximum epochs for final training.
            seed: Random seed.

        Returns:
            Tuple of (trained model, best_params dict).
        """
        best_val_loss = float("inf")
        best_params = None

        for trial in range(self.num_trials):
            batch_size = int(np.random.choice(BATCH_SIZE_OPTIONS))
            lstm_pair = LSTM_UNITS_OPTIONS[np.random.randint(len(LSTM_UNITS_OPTIONS))]
            dropout_pair = DROPOUT_OPTIONS[np.random.randint(len(DROPOUT_OPTIONS))]
            tuning_epochs = int(np.random.choice(TUNING_EPOCHS_OPTIONS))
            lr = float(np.random.choice(LR_OPTIONS))

            params = {
                "lstm_units_1": lstm_pair[0],
                "lstm_units_2": lstm_pair[1],
                "dropout_1": dropout_pair[0],
                "dropout_2": dropout_pair[1],
                "lr": lr,
                "batch_size": batch_size,
                "tuning_epochs": tuning_epochs,
            }

            logger.info(
                f"Trial {trial + 1}/{self.num_trials} {model_name}: "
                f"bs={batch_size}, units=({lstm_pair[0]},{lstm_pair[1]}), "
                f"drop=({dropout_pair[0]},{dropout_pair[1]}), lr={lr:.0e}, ep={tuning_epochs}"
            )
            tf.random.set_seed(seed + trial)

            m = build_model(
                num_words=self.num_words,
                embed_dim=self.embed_dim,
                embedding_matrix=self.embedding_matrix,
                use_bidirectional=use_bidirectional,
                lstm_units_1=lstm_pair[0],
                lstm_units_2=lstm_pair[1],
                dropout_1=dropout_pair[0],
                dropout_2=dropout_pair[1],
                lr=lr,
                embed_trainable=self.embed_trainable,
                recurrent_dropout=self.recurrent_dropout,
                gradient_clip_norm=self.gradient_clip_norm,
            )

            ckpt_path = os.path.join(tempfile.gettempdir(), f"best_{model_name}_trial{trial}.keras")
            cb = [
                ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
                ),
                ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=0),
            ]
            h = m.fit(
                X_train,
                y_train,
                validation_split=self.val_split,
                epochs=tuning_epochs,
                batch_size=batch_size,
                callbacks=cb,
                class_weight=self.class_weight,
                verbose=0,
            )
            val_loss = min(h.history["val_loss"])
            val_acc = max(h.history["val_accuracy"])
            logger.info(f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params

        logger.info(
            f"Best params for {model_name}: bs={best_params['batch_size']}, "
            f"units=({best_params['lstm_units_1']},{best_params['lstm_units_2']}), "
            f"drop=({best_params['dropout_1']},{best_params['dropout_2']}), "
            f"lr={best_params['lr']:.0e}, ep={best_params['tuning_epochs']} "
            f"(val_loss={best_val_loss:.4f})"
        )

        # Train final model with best params
        logger.info(f"Training final {model_name}")
        tf.random.set_seed(seed)

        final_model = build_model(
            num_words=self.num_words,
            embed_dim=self.embed_dim,
            embedding_matrix=self.embedding_matrix,
            use_bidirectional=use_bidirectional,
            lstm_units_1=best_params["lstm_units_1"],
            lstm_units_2=best_params["lstm_units_2"],
            dropout_1=best_params["dropout_1"],
            dropout_2=best_params["dropout_2"],
            lr=best_params["lr"],
            embed_trainable=self.embed_trainable,
            recurrent_dropout=self.recurrent_dropout,
            gradient_clip_norm=self.gradient_clip_norm,
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_patience,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=self.lr_reduce_patience,
                min_lr=self.min_lr,
            ),
            ModelCheckpoint(
                os.path.join(tempfile.gettempdir(), f"best_{model_name}_final.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ]

        final_model.fit(
            X_train,
            y_train,
            validation_split=self.val_split,
            epochs=max_epochs,
            batch_size=best_params["batch_size"],
            callbacks=callbacks,
            class_weight=self.class_weight,
            verbose=1,
        )

        return final_model, best_params
