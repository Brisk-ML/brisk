"""Unit test dataset plots."""
import numpy as np
import pandas as pd
import pytest

from brisk.evaluation.evaluators.builtin import dataset_plots
from brisk.theme import plot_settings

class TestHistogram:
    @pytest.fixture
    def mock_evaluator(self):
        evaluator = dataset_plots.Histogram(
            method_name="test_histogram",
            description="test description",
            plot_settings=plot_settings.PlotSettings()
        )
        return evaluator
    
    def test_generate_plot_data(self, mock_evaluator):
        """Test that generate_plot_data returns correct structure."""
        train_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name='feature1')
        test_data = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5], name='feature1')
        
        result = mock_evaluator.generate_plot_data(
            train_data=train_data,
            test_data=test_data,
            feature_name='feature1'
        )
        
        pd.testing.assert_series_equal(result["train_series"], train_data)
        pd.testing.assert_series_equal(result["test_series"], test_data)
    
class TestBarPlot:
    @pytest.fixture
    def mock_evaluator(self):
        evaluator = dataset_plots.BarPlot(
            method_name="test_barplot",
            description="test description",
            plot_settings=plot_settings.PlotSettings()
        )
        return evaluator
    
    def test_generate_plot_data_value_counts(self, mock_evaluator):
        train_data = pd.Series(['A', 'A', 'B', 'B', 'C'])
        test_data = pd.Series(['A', 'B', 'B', 'C', 'C'])
        
        result = mock_evaluator.generate_plot_data(
            train_data=train_data,
            test_data=test_data,
            feature_name='category'
        )
        
        expected_train = pd.Series({'A': 2, 'B': 2, 'C': 1}, name="count")
        pd.testing.assert_series_equal(
            result["train_value_counts"].sort_index(),
            expected_train.sort_index()
        )

        expected_test = pd.Series({'A': 1, 'B': 2, 'C': 2}, name="count")
        pd.testing.assert_series_equal(
            result["test_value_counts"].sort_index(),
            expected_test.sort_index()
        )
        
        assert result["feature_name"] == 'category'
    
    def test_generate_plot_data_different_categories(self, mock_evaluator):
        """Test handling when train and test have different categories."""
        train_data = pd.Series(['A', 'A', 'B', 'B', 'C'])
        test_data = pd.Series(['B', 'C', 'C', 'D', 'D'])
        
        result = mock_evaluator.generate_plot_data(
            train_data=train_data,
            test_data=test_data,
            feature_name='category'
        )
        
        train_counts = result["train_value_counts"]
        assert 'A' in train_counts.index
        assert 'B' in train_counts.index
        assert 'C' in train_counts.index
        assert 'D' not in train_counts.index
        
        test_counts = result["test_value_counts"]
        assert 'A' not in test_counts.index
        assert 'B' in test_counts.index
        assert 'C' in test_counts.index
        assert 'D' in test_counts.index
    
    def test_generate_plot_data_single_category(self, mock_evaluator):
        """Test with data containing only one category."""
        train_data = pd.Series(['A', 'A', 'A', 'A', 'A'])
        test_data = pd.Series(['A', 'A', 'A'])
        
        result = mock_evaluator.generate_plot_data(
            train_data=train_data,
            test_data=test_data,
            feature_name='single_cat'
        )
        
        assert len(result["train_value_counts"]) == 1
        assert result["train_value_counts"]['A'] == 5
        assert len(result["test_value_counts"]) == 1
        assert result["test_value_counts"]['A'] == 3
    
    def test_generate_plot_data_empty_category_values(self, mock_evaluator):
        """Test with categories that appear in one split but not the other."""
        train_data = pd.Series(['X', 'Y', 'Z'])
        test_data = pd.Series(['A', 'B', 'C'])
        
        result = mock_evaluator.generate_plot_data(
            train_data=train_data,
            test_data=test_data,
            feature_name='disjoint'
        )
        
        # Verify no overlap
        train_categories = set(result["train_value_counts"].index)
        test_categories = set(result["test_value_counts"].index)
        assert train_categories.isdisjoint(test_categories)
        
        # Verify counts are preserved
        assert result["train_value_counts"]['X'] == 1
        assert result["test_value_counts"]['A'] == 1


class TestCorrelationMatrix:
    @pytest.fixture
    def mock_evaluator(self):
        evaluator = dataset_plots.CorrelationMatrix(
            method_name="test_correlation",
            description="test description",
            plot_settings=plot_settings.PlotSettings()
        )
        return evaluator
    
    def test_generate_plot_data_correlation_matrix(self, mock_evaluator):
        """Test that correlation matrix is correctly calculated."""
        np.random.seed(42)
        train_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'feature3': [5, 4, 3, 2, 1]
        })
        
        result = mock_evaluator.generate_plot_data(
            train_data=train_data,
            continuous_features=['feature1', 'feature2', 'feature3']
        )
        
        corr_matrix = result["correlation_matrix"]
        assert corr_matrix.shape == (3, 3)
        
        np.testing.assert_array_almost_equal(
            np.diag(corr_matrix.values), [1.0, 1.0, 1.0]
        )
        
        assert np.isclose(corr_matrix.loc['feature1', 'feature2'], 1.0)
        assert np.isclose(corr_matrix.loc['feature2', 'feature1'], 1.0)
        
        assert np.isclose(corr_matrix.loc['feature1', 'feature3'], -1.0)
        assert np.isclose(corr_matrix.loc['feature3', 'feature1'], -1.0)
    
    def test_generate_plot_data_plot_dimensions(self, mock_evaluator):
        """Test that plot dimensions scale with number of features."""
        train_data = pd.DataFrame({
            f'feature{i}': np.random.randn(10) for i in range(5)
        })
        
        result_5 = mock_evaluator.generate_plot_data(
            train_data=train_data,
            continuous_features=[f'feature{i}' for i in range(5)]
        )
        
        assert result_5["width"] == max(12, 0.5 * 5)
        assert result_5["height"] == max(8, 0.5 * 5 * 0.75)
    
    def test_generate_plot_data_minimum_dimensions(self, mock_evaluator):
        """Test that minimum plot dimensions are enforced."""
        train_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        result = mock_evaluator.generate_plot_data(
            train_data=train_data,
            continuous_features=['feature1', 'feature2']
        )
        
        assert result["width"] == 12  # Minimum enforced
        assert result["height"] == 8  # Minimum enforced
    
    def test_generate_plot_data_subset_of_features(self, mock_evaluator):
        """Test that only specified features are included in correlation."""
        train_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'feature3': [5, 4, 3, 2, 1],
            'feature4': [10, 20, 30, 40, 50],
            'feature5': [5, 10, 15, 20, 25]
        })
        
        result = mock_evaluator.generate_plot_data(
            train_data=train_data,
            continuous_features=['feature1', 'feature2', 'feature3']
        )
        
        corr_matrix = result["correlation_matrix"]
        
        assert corr_matrix.shape == (3, 3)
        assert 'feature4' not in corr_matrix.columns
        assert 'feature5' not in corr_matrix.columns
    
    def test_generate_plot_data_uncorrelated_features(self, mock_evaluator):
        """Test correlation matrix with uncorrelated features."""
        np.random.seed(42)
        train_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        
        result = mock_evaluator.generate_plot_data(
            train_data=train_data,
            continuous_features=['feature1', 'feature2', 'feature3']
        )
        
        corr_matrix = result["correlation_matrix"]
        
        off_diagonal = corr_matrix.values[~np.eye(3, dtype=bool)]
        assert np.all(np.abs(off_diagonal) < 0.5)
    
    def test_generate_plot_data_single_feature(self, mock_evaluator):
        """Test correlation matrix with single feature."""
        train_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5]
        })
        
        result = mock_evaluator.generate_plot_data(
            train_data=train_data,
            continuous_features=['feature1']
        )
        
        corr_matrix = result["correlation_matrix"]
        
        assert corr_matrix.shape == (1, 1)
        assert np.isclose(corr_matrix.loc['feature1', 'feature1'], 1.0)

