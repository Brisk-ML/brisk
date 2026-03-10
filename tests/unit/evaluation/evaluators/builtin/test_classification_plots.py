"""Unit tests for classification plots."""
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn import linear_model
from sklearn import svm

from brisk.evaluation.evaluators.builtin import classification_plots
from brisk.theme import plot_settings

class MockModel:
    """Base mock model for testing."""
    def __init__(self, prediction_method='predict_proba'):
        self.prediction_method = prediction_method
        self.display_name = "Mock Model"
        self.wrapper_name = "mock_wrapper"

    def predict(self, X):
        return (X[:, 0] > 0.5).astype(int)

    def predict_proba(self, X):
        if self.prediction_method != 'predict_proba':
            raise AttributeError("Model doesn't have predict_proba")
        proba_positive = X[:, 0]
        proba_negative = 1 - proba_positive
        return np.column_stack([proba_negative, proba_positive])

    def decision_function(self, X):
        if self.prediction_method != 'decision_function':
            raise AttributeError("Model doesn't have decision_function")
        return 2 * X[:, 0] - 1


@pytest.mark.unit
class TestPlotConfusionHeatmap:
    @pytest.fixture
    def sample_data(self):
        y_true = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        predictions = pd.Series([0, 1, 1, 1, 2, 2, 2, 2, 0, 1])
        return predictions, y_true

    def test_generate_plot_data(self, sample_data):
        """Test generate_plot_data returns correct confusion matrix data."""

        predictions, y_true = sample_data
        evaluator = classification_plots.PlotConfusionHeatmap(
            method_name="test",
            description="test",
            plot_settings=plot_settings.PlotSettings()
        )

        result = evaluator.generate_plot_data(predictions, y_true)

        # True=0, Pred=0: 1 correct prediction (10%)
        cell_00 = result[
            (result["True Label"] == 0) &
            (result["Predicted Label"] == 0)
        ].iloc[0]
        assert cell_00["Percentage"] == 10.0
        assert "1\n(10.0%)" in cell_00["Label"]

        # True=1, Pred=1: 2 correct predictions (20%)
        cell_11 = result[
            (result["True Label"] == 1) &
            (result["Predicted Label"] == 1)
        ].iloc[0]
        assert cell_11["Percentage"] == 20.0
        assert "2\n(20.0%)" in cell_11["Label"]

        # True=2, Pred=2: 3 correct predictions (30%)
        cell_22 = result[
            (result["True Label"] == 2) &
            (result["Predicted Label"] == 2)
        ].iloc[0]
        assert cell_22["Percentage"] == 30.0
        assert "3\n(30.0%)" in cell_22["Label"]

        # Verify percentages sum to 100%
        assert np.isclose(result["Percentage"].sum(), 100.0)

    def test_generate_plot_data_one_label(self):
        """Test generate_plot_data with only one class label."""

        y_true = np.array([1, 1, 1, 1, 1])
        predictions = pd.Series([1, 1, 1, 1, 1])

        evaluator = classification_plots.PlotConfusionHeatmap(
            method_name="test",
            description="test",
            plot_settings=plot_settings.PlotSettings()
        )

        result = evaluator.generate_plot_data(predictions, y_true)

        assert result.iloc[0]["Percentage"] == 100.0
        assert result.iloc[0]["True Label"] == 1
        assert result.iloc[0]["Predicted Label"] == 1
        assert "5\n(100.0%)" in result.iloc[0]["Label"]


@pytest.mark.unit
class TestPlotRocCurve:
    @pytest.fixture
    def binary_data(self):
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_generate_plot_data_roc_data(self, binary_data):
        X, y = binary_data
        model = linear_model.LogisticRegression(random_state=42)
        model.fit(X, y)

        evaluator = classification_plots.PlotRocCurve(
            method_name="test",
            description="test",
            plot_settings=plot_settings.PlotSettings()
        )

        plot_data, auc_data, auc = evaluator.generate_plot_data(
            model, X, y, pos_label=1
        )

        # ROC Curve should start at (0,0) and end at (1,1)
        roc_curve_data = plot_data[plot_data["Type"] == "ROC Curve"]
        assert roc_curve_data.iloc[0]["False Positive Rate"] == 0.0
        assert roc_curve_data.iloc[0]["True Positive Rate"] == 0.0
        assert roc_curve_data.iloc[-1]["False Positive Rate"] == 1.0
        assert roc_curve_data.iloc[-1]["True Positive Rate"] == 1.0

        # Random guessing should be diagonal line
        random_data = plot_data[plot_data["Type"] == "Random Guessing"]
        assert len(random_data) == 2
        assert list(random_data["False Positive Rate"]) == [0, 1]
        assert list(random_data["True Positive Rate"]) == [0, 1]

    def test_generate_plot_data_auc_data(self, binary_data):
        X, y = binary_data
        model = linear_model.LogisticRegression(random_state=42)
        model.fit(X, y)

        evaluator = classification_plots.PlotRocCurve(
            method_name="test",
            description="test",
            plot_settings=plot_settings.PlotSettings()
        )

        plot_data, auc_data, auc = evaluator.generate_plot_data(
            model, X, y, pos_label=1
        )

        # FPR should be evenly spaced from 0 to 1
        fpr = auc_data["False Positive Rate"].values
        assert np.isclose(fpr[0], 0.0)
        assert np.isclose(fpr[-1], 1.0)
        assert np.all(np.diff(fpr) > 0)  # Monotonically increasing

        # TPR should be monotonically increasing
        tpr = auc_data["True Positive Rate"].values
        assert np.all(np.diff(tpr) >= 0)

    def test_generate_plot_data_predict_proba(self, binary_data):
        """Test correct ROC Curve when using predict_proba()."""
        X, y = binary_data
        model = linear_model.LogisticRegression(random_state=42)
        model.fit(X, y)

        assert hasattr(model, 'predict_proba')

        evaluator = classification_plots.PlotRocCurve(
            method_name="test",
            description="test",
            plot_settings=plot_settings.PlotSettings()
        )

        plot_data, auc_data, auc = evaluator.generate_plot_data(
            model, X, y, pos_label=1
        )

        roc_curve_data = plot_data[plot_data["Type"] == "ROC Curve"]
        assert len(roc_curve_data) > 2  # More than just endpoints

        # FPR and TPR should be in valid range
        assert roc_curve_data["False Positive Rate"].between(0, 1).all()
        assert roc_curve_data["True Positive Rate"].between(0, 1).all()

    def test_generate_plot_data_decision_function(self, binary_data):
        X, y = binary_data
        model = svm.SVC(kernel='linear', random_state=42)
        model.fit(X, y)

        evaluator = classification_plots.PlotRocCurve(
            method_name="test",
            description="test",
            plot_settings=plot_settings.PlotSettings()
        )

        plot_data, auc_data, auc = evaluator.generate_plot_data(
            model, X, y, pos_label=1
        )

        roc_curve_data = plot_data[plot_data["Type"] == "ROC Curve"]
        assert len(roc_curve_data) > 2
        assert roc_curve_data["False Positive Rate"].between(0, 1).all()
        assert roc_curve_data["True Positive Rate"].between(0, 1).all()

    def test_generate_plot_data_predict(self, binary_data):
        """Test correct ROC Curve when using predict()."""
        X, y = binary_data
        model = Mock()
        model.predict = lambda X: (X[:, 0] + X[:, 1] > 0).astype(int)

        del model.predict_proba
        del model.decision_function

        evaluator = classification_plots.PlotRocCurve(
            method_name="test",
            description="test",
            plot_settings=plot_settings.PlotSettings()
        )

        plot_data, auc_data, auc = evaluator.generate_plot_data(
            model, X, y, pos_label=1
        )

        assert 0.0 <= auc <= 1.0
        roc_curve_data = plot_data[plot_data["Type"] == "ROC Curve"]
        assert len(roc_curve_data) >= 2  # At least endpoints


@pytest.mark.unit
class TestPlotPrecisionRecallCurve:
    @pytest.fixture
    def binary_data(self):
        """Create binary classification test data."""
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_generate_plot_data_pr_curve(self, binary_data):
        X, y = binary_data
        model = linear_model.LogisticRegression(random_state=42)
        model.fit(X, y)

        evaluator = classification_plots.PlotPrecisionRecallCurve(
            method_name="test",
            description="test",
            plot_settings=plot_settings.PlotSettings()
        )

        plot_data, ap_score = evaluator.generate_plot_data(
            model, X, y, pos_label=1
        )

        pr_curve_data = plot_data[plot_data["Type"] == "PR Curve"]
        assert len(pr_curve_data) > 2
        assert pr_curve_data["Precision"].between(0, 1).all()
        assert pr_curve_data["Recall"].between(0, 1).all()

    def test_generate_plot_data_ap_line(self, binary_data):
        """Test that AP score reference line is correct."""
        X, y = binary_data
        model = linear_model.LogisticRegression(random_state=42)
        model.fit(X, y)

        evaluator = classification_plots.PlotPrecisionRecallCurve(
            method_name="test",
            description="test",
            plot_settings=plot_settings.PlotSettings()
        )

        plot_data, ap_score = evaluator.generate_plot_data(
            model, X, y, pos_label=1
        )

        # Find AP score line
        ap_line = plot_data[plot_data["Type"].str.contains("AP Score")]

        assert len(ap_line) == 2
        assert list(ap_line["Recall"]) == [0, 1]

        precision_values = ap_line["Precision"].values
        assert np.isclose(precision_values[0], ap_score)
        assert np.isclose(precision_values[1], ap_score)

    def test_generate_plot_data_predict_proba(self, binary_data):
        """Test correct PR curve when using predict_proba()."""
        X, y = binary_data
        model = linear_model.LogisticRegression(random_state=42)
        model.fit(X, y)

        assert hasattr(model, 'predict_proba')

        evaluator = classification_plots.PlotPrecisionRecallCurve(
            method_name="test",
            description="test",
            plot_settings=plot_settings.PlotSettings()
        )

        plot_data, ap_score = evaluator.generate_plot_data(
            model, X, y, pos_label=1
        )

        pr_curve_data = plot_data[plot_data["Type"] == "PR Curve"]
        assert len(pr_curve_data) > 2
        assert pr_curve_data["Precision"].between(0, 1).all()
        assert pr_curve_data["Recall"].between(0, 1).all()

    def test_generate_plot_data_decision_function(self, binary_data):
        """Test correct PR curve when using decision_function()."""
        X, y = binary_data
        model = svm.SVC(kernel='linear', random_state=42)
        model.fit(X, y)

        evaluator = classification_plots.PlotPrecisionRecallCurve(
            method_name="test",
            description="test",
            plot_settings=plot_settings.PlotSettings()
        )

        plot_data, ap_score = evaluator.generate_plot_data(
            model, X, y, pos_label=1
        )

        pr_curve_data = plot_data[plot_data["Type"] == "PR Curve"]
        assert len(pr_curve_data) > 2
        assert pr_curve_data["Precision"].between(0, 1).all()
        assert pr_curve_data["Recall"].between(0, 1).all()

    def test_generate_plot_data_predict(self, binary_data):
        """Test correct PR curve when using predict()."""
        X, y = binary_data
        model = Mock()
        model.predict = lambda X: (X[:, 0] + X[:, 1] > 0).astype(int)

        del model.predict_proba
        del model.decision_function

        evaluator = classification_plots.PlotPrecisionRecallCurve(
            method_name="test",
            description="test",
            plot_settings=plot_settings.PlotSettings()
        )

        plot_data, ap_score = evaluator.generate_plot_data(
            model, X, y, pos_label=1
        )

        pr_curve_data = plot_data[plot_data["Type"] == "PR Curve"]
        assert len(pr_curve_data) >= 2
