from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder


class ModelEvaluator:
    """Evaluate classification model performance."""

    def __init__(self, label_encoder: LabelEncoder):
        self.label_encoder = label_encoder

    def evaluate(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_test_text: Optional[List[str]] = None,
    ) -> Dict:
        """Evaluate model and return metrics.

        Args:
            model: Trained Keras model.
            X_test: Test padded sequences.
            y_test: Test labels (integer encoded).
            X_test_text: Optional original text for prediction table.

        Returns:
            Dictionary with eval_table, report_df, pred_table, and metrics.
        """
        y_prob = model.predict(X_test).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=1
        )
        auc = roc_auc_score(y_test, y_prob)

        eval_table = pd.DataFrame(
            [
                {"metric": "accuracy", "value": acc},
                {"metric": "precision", "value": prec},
                {"metric": "recall", "value": rec},
                {"metric": "f1_score", "value": f1},
                {"metric": "roc_auc", "value": auc},
            ]
        )

        target_names = list(self.label_encoder.classes_)
        report_dict = classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True, zero_division=1
        )
        report_df = (
            pd.DataFrame(report_dict)
            .transpose()
            .reset_index()
            .rename(columns={"index": "label"})
        )

        pred_table = None
        if X_test_text is not None:
            pred_table = pd.DataFrame(
                {
                    "text": X_test_text,
                    "label_actual": self.label_encoder.inverse_transform(y_test),
                    "label_pred": self.label_encoder.inverse_transform(y_pred),
                    "score_unsafe": y_prob,
                }
            )

        return {
            "eval_table": eval_table,
            "report_df": report_df,
            "pred_table": pred_table,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "y_prob": y_prob,
            "y_pred": y_pred,
        }

    @staticmethod
    def compute_class_weights(y_train: np.ndarray) -> Optional[Dict[int, float]]:
        """Compute balanced class weights.

        Args:
            y_train: Training labels (integer encoded).

        Returns:
            Dictionary mapping class index to weight, or None if balanced.
        """
        from sklearn.utils import class_weight

        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            return None

        cw = class_weight.compute_class_weight(
            class_weight="balanced", classes=unique_classes, y=y_train
        )
        return dict(enumerate(cw))
