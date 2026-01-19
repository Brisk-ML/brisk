"""Unit tests for classifiaction measures."""

import pytest
import numpy as np
import pandas as pd

from brisk.evaluation.evaluators.builtin import classification_measures

@pytest.fixture
def confusion_matrix_evaluator():
    """Create a ConfusionMatrix evaluator instance."""
    evaluator = classification_measures.ConfusionMatrix(
        method_name="confusion_matrix",
        description="Confusion matrix for classification"
    )
    return evaluator


class TestConfusionMatrix:
    def test_calculate_measures(self, confusion_matrix_evaluator):
        """Test confusion matrix calculation with normal multi-class data."""
        y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
        predictions = pd.Series([0, 0, 1, 2, 2, 1, 0, 1, 2])
        
        results = confusion_matrix_evaluator.calculate_measures(
            predictions, y_true
        )
        
        assert results["labels"] == [0, 1, 2]
        
        cm = results["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)
        
    def test_calculate_measures_empty_array(self):
        """Test that empty arrays raise an appropriate error."""
        evaluator = classification_measures.ConfusionMatrix(
            "confusion_matrix",
            "Confusion matrix for classification"
        )
        y_true = np.array([])
        predictions = pd.Series([])
        
        with pytest.raises((ValueError, IndexError)):
            evaluator.calculate_measures(predictions, y_true)

    def test_calculate_measures_one_unique_label(self):
        """Test confusion matrix with only one class present."""
        evaluator = classification_measures.ConfusionMatrix(
            "confusion_matrix",
            "Confusion matrix for classification"
        )
        y_true = np.array([1, 1, 1, 1, 1])
        predictions = pd.Series([1, 1, 1, 1, 1])
        
        results = evaluator.calculate_measures(predictions, y_true)
        
        assert results["labels"] == [1]
        cm = results["confusion_matrix"]
        assert len(cm) == 1
        assert len(cm[0]) == 1
