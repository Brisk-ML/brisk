"""Metric configuration for binary classification e2e tests."""
import numpy as np
from sklearn import metrics as sk_metrics

import brisk


def huber_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Custom Huber loss metric for testing."""
    DELTA = 1
    loss = np.where(
        np.abs(y_true - y_pred) <= DELTA,
        0.5 * (y_true - y_pred)**2,
        DELTA * (np.abs(y_true - y_pred) - 0.5 * DELTA)
    )
    return np.mean(loss)


def fake_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_metadata: dict
) -> float:
    """Custom metric that uses split_metadata for testing."""
    return np.mean(
        (y_true - y_pred) / 
        (split_metadata["num_features"] / split_metadata["num_samples"])
    )


METRIC_CONFIG = brisk.MetricManager(
    *brisk.REGRESSION_METRICS,
    *brisk.CLASSIFICATION_METRICS,
    brisk.MetricWrapper(
        name="huber_loss",
        func=huber_loss,
        display_name="Huber Loss",
        greater_is_better=False
    ),
    brisk.MetricWrapper(
        name="fake_metric",
        func=fake_metric,
        display_name="Fake Metric",
        greater_is_better=False
    ),
    brisk.MetricWrapper(
        name="f1_multiclass",
        func=sk_metrics.f1_score,
        display_name="F1 Score (Multiclass)",
        average="weighted",
        greater_is_better=True
    ),
    brisk.MetricWrapper(
        name="precision_multiclass",
        func=sk_metrics.precision_score,
        display_name="Precision (Multiclass)",
        average="micro",
        greater_is_better=True
    ),
    brisk.MetricWrapper(
        name="recall_multiclass",
        func=sk_metrics.recall_score,
        display_name="Recall (Multiclass)",
        average="macro",
        greater_is_better=True
    ),
)
