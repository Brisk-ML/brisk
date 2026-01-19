"""Unit tests for regression plots."""
import pandas as pd
import pytest

from brisk.evaluation.evaluators.builtin import regression_plots
from brisk.theme import plot_settings


class TestPlotPredVsObs:
    @pytest.fixture
    def mock_evaluator(self):
        """Create a PlotPredVsObs evaluator."""
        
        evaluator = regression_plots.PlotPredVsObs(
            method_name="test_pred_vs_obs",
            description="test description",
            plot_settings=plot_settings.PlotSettings()
        )
        
        return evaluator
    
    def test_generate_plot_data_basic(self, mock_evaluator):
        """Test basic functionality with simple data."""
        predictions = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_true = pd.Series([1.1, 2.2, 2.9, 4.1, 5.3])
        
        plot_data, max_range = mock_evaluator.generate_plot_data(
            prediction=predictions,
            y_true=y_true
        )
        
        expected_max = max(y_true.max(), predictions.max())
        assert max_range == expected_max
        assert max_range == 5.3
    
    def test_generate_plot_data_negative_values(self, mock_evaluator):
        """Test with negative values in predictions and observations."""
        predictions = pd.Series([-5.0, 0.0, 5.0])
        y_true = pd.Series([-3.0, 1.0, 4.0])
        
        plot_data, max_range = mock_evaluator.generate_plot_data(
            prediction=predictions,
            y_true=y_true
        )
        
        assert max_range == 5.0
    
    def test_generate_plot_data_single_point(self, mock_evaluator):
        """Test with single data point."""
        predictions = pd.Series([3.5])
        y_true = pd.Series([3.0])
        
        plot_data, max_range = mock_evaluator.generate_plot_data(
            prediction=predictions,
            y_true=y_true
        )
        
        assert len(plot_data) == 1
        assert max_range == 3.5


class TestPlotResiduals:
    @pytest.fixture
    def mock_evaluator(self):
        evaluator = regression_plots.PlotResiduals(
            method_name="test_residuals",
            description="test description",
            plot_settings=plot_settings.PlotSettings()
        )
        return evaluator
    
    def test_generate_plot_data_basic(self, mock_evaluator):
        """Test basic residual calculation."""
        predictions = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_true = pd.Series([1.5, 2.5, 2.8, 4.2, 5.1])
        
        plot_data = mock_evaluator.generate_plot_data(
            predictions=predictions,
            y=y_true
        )
        
        expected_residuals = y_true - predictions
        pd.testing.assert_series_equal(
            plot_data["Residual (Observed - Predicted)"],
            expected_residuals,
            check_names=False
        )
    
    def test_generate_plot_data_perfect_predictions(self, mock_evaluator):
        """Test when predictions are perfect (residuals = 0)."""
        predictions = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        plot_data = mock_evaluator.generate_plot_data(
            predictions=predictions,
            y=y_true
        )
        
        # All residuals should be 0
        assert all(plot_data["Residual (Observed - Predicted)"] == 0)
    
    def test_generate_plot_data_single_point(self, mock_evaluator):
        """Test with single data point."""
        predictions = pd.Series([3.5])
        y_true = pd.Series([4.0])
        
        plot_data = mock_evaluator.generate_plot_data(
            predictions=predictions,
            y=y_true
        )
        
        assert len(plot_data) == 1
        assert plot_data["Residual (Observed - Predicted)"].iloc[0] == 0.5
