from typing import Optional

import tensorflow as tf
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    LSTM,
)
from tensorflow.keras.models import Sequential


def build_model(
    num_words: int,
    embed_dim: int,
    embedding_matrix,
    use_bidirectional: bool,
    lstm_units_1: int,
    lstm_units_2: int,
    dropout_1: float,
    dropout_2: float,
    lr: float,
    dense_units: int = 64,
    dropout_dense: float = 0.2,
    embed_trainable: bool = False,
    recurrent_dropout: float = 0.0,
    gradient_clip_norm: float = 1.0,
) -> Sequential:
    """Build BiLSTM or LSTM model for binary classification.

    Args:
        num_words: Vocabulary size for embedding layer.
        embed_dim: Embedding dimension.
        embedding_matrix: Pre-trained embedding matrix.
        use_bidirectional: If True, use Bidirectional LSTM layers.
        lstm_units_1: Units in first LSTM layer.
        lstm_units_2: Units in second LSTM layer.
        dropout_1: Dropout rate after first LSTM layer.
        dropout_2: Dropout rate after second LSTM layer.
        lr: Learning rate.
        dense_units: Units in dense layer.
        dropout_dense: Dropout rate after dense layer.
        embed_trainable: Whether embedding layer is trainable.
        recurrent_dropout: Recurrent dropout for LSTM.
        gradient_clip_norm: Gradient clipping norm.

    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential()
    model.add(
        Embedding(
            input_dim=num_words,
            output_dim=embed_dim,
            weights=[embedding_matrix],
            trainable=embed_trainable,
        )
    )

    lstm_kwargs = dict(return_sequences=True, recurrent_dropout=recurrent_dropout)
    lstm_kwargs2 = dict(recurrent_dropout=recurrent_dropout)

    if use_bidirectional:
        model.add(Bidirectional(LSTM(lstm_units_1, **lstm_kwargs)))
    else:
        model.add(LSTM(lstm_units_1, **lstm_kwargs))
    model.add(Dropout(dropout_1))

    if use_bidirectional:
        model.add(Bidirectional(LSTM(lstm_units_2, **lstm_kwargs2)))
    else:
        model.add(LSTM(lstm_units_2, **lstm_kwargs2))
    model.add(Dropout(dropout_2))

    model.add(Dense(dense_units, activation="relu", kernel_constraint=MaxNorm(3)))
    model.add(Dropout(dropout_dense))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr, clipnorm=gradient_clip_norm
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model
