"""Unit tests for dataset measure evaluators."""

import numpy as np
import pandas as pd
import pytest

from brisk.evaluation.evaluators.builtin import dataset_measures 

class TestContinuousStatistics:
    @pytest.fixture
    def mock_evaluator(self):
        evaluator = dataset_measures.ContinuousStatistics(
            method_name="test_continuous_stats",
            description="test description"
        )
        return evaluator
    
    @pytest.fixture
    def sample_continuous_data(self):
        np.random.seed(42)
        train_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'feature3': [0.5, 1.5, 2.5, 3.5, 4.5]
        })
        
        test_data = pd.DataFrame({
            'feature1': [1.5, 2.5, 3.5, 4.5, 5.5],
            'feature2': [15.0, 25.0, 35.0, 45.0, 55.0],
            'feature3': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        return train_data, test_data
    
    def test_calculate_measures_single_feature(self, mock_evaluator, sample_continuous_data):
        """Test calculate_measures with a single feature."""
        train_data, test_data = sample_continuous_data
        
        results = mock_evaluator.calculate_measures(
            train_data=train_data,
            test_data=test_data,
            feature_names=['feature1']
        )
        
        train_stats = results['feature1']['train']
        assert np.isclose(train_stats['mean'], 3.0)
        assert np.isclose(train_stats['median'], 3.0)
        assert np.isclose(train_stats['min'], 1.0)
        assert np.isclose(train_stats['max'], 5.0)
        assert np.isclose(train_stats['range'], 4.0)
        assert np.isclose(train_stats['25_percentile'], 2.0)
        assert np.isclose(train_stats['75_percentile'], 4.0)
    
    def test_calculate_measures_multiple_features(self, mock_evaluator, sample_continuous_data):
        """Test calculate_measures with multiple features."""
        train_data, test_data = sample_continuous_data
        
        results = mock_evaluator.calculate_measures(
            train_data=train_data,
            test_data=test_data,
            feature_names=['feature1', 'feature2', 'feature3']
        )
        
        for feature in ['feature1', 'feature2', 'feature3']:
            assert 'train' in results[feature]
            assert 'test' in results[feature]
    
    def test_calculate_measures_statistics_accuracy(self, mock_evaluator):
        """Test that calculated statistics are mathematically correct."""
        train_data = pd.DataFrame({
            'uniform': [1.0, 2.0, 3.0, 4.0, 5.0],  # Uniform distribution
            'constant': [10.0, 10.0, 10.0, 10.0, 10.0]  # No variance
        })
        
        test_data = pd.DataFrame({
            'uniform': [1.0, 2.0, 3.0, 4.0, 5.0],
            'constant': [10.0, 10.0, 10.0, 10.0, 10.0]
        })
        
        results = mock_evaluator.calculate_measures(
            train_data=train_data,
            test_data=test_data,
            feature_names=['uniform', 'constant']
        )
        
        uniform_stats = results['uniform']['train']
        assert np.isclose(uniform_stats['mean'], 3.0)
        assert np.isclose(uniform_stats['variance'], 2.5)
        assert np.isclose(uniform_stats['skewness'], 0.0, atol=1e-10)
        
        constant_stats = results['constant']['train']
        assert np.isclose(constant_stats['mean'], 10.0)
        assert np.isclose(constant_stats['variance'], 0.0)
        assert np.isclose(constant_stats['std_dev'], 0.0)
        assert np.isclose(constant_stats['range'], 0.0)
    
    def test_calculate_measures_coefficient_of_variation(self, mock_evaluator):
        """Test coefficient of variation calculation and division by zero handling."""
        train_data = pd.DataFrame({
            'normal': [1.0, 2.0, 3.0, 4.0, 5.0],
            'zero_mean': [-2.0, -1.0, 0.0, 1.0, 2.0]  # Mean = 0
        })
        
        test_data = pd.DataFrame({
            'normal': [1.5, 2.5, 3.5, 4.5, 5.5],
            'zero_mean': [-2.5, -1.5, -0.5, 0.5, 1.5]
        })
        
        results = mock_evaluator.calculate_measures(
            train_data=train_data,
            test_data=test_data,
            feature_names=['normal', 'zero_mean']
        )
        
        normal_cov = results['normal']['train']['coefficient_of_variation']
        assert np.isfinite(normal_cov)
        
        zero_mean_cov = results['zero_mean']['train']['coefficient_of_variation']
        assert zero_mean_cov is None
    
    def test_calculate_measures_empty_feature_list(self, mock_evaluator, sample_continuous_data):
        """Test calculate_measures with empty feature list."""
        train_data, test_data = sample_continuous_data
        
        results = mock_evaluator.calculate_measures(
            train_data=train_data,
            test_data=test_data,
            feature_names=[]
        )
        
        assert isinstance(results, dict)
        assert len(results) == 0
    
    def test_calculate_measures_skewness_and_kurtosis(self, mock_evaluator):
        """Test skewness and kurtosis calculations."""
        train_data = pd.DataFrame({
            'right_skewed': [1, 1, 1, 2, 2, 3, 4, 5, 10, 20]
        })
        
        test_data = pd.DataFrame({
            'right_skewed': [1, 1, 1, 2, 2, 3, 4, 5, 10, 20]
        })
        
        results = mock_evaluator.calculate_measures(
            train_data=train_data,
            test_data=test_data,
            feature_names=['right_skewed']
        )
        
        skewness = results['right_skewed']['train']['skewness']
        assert skewness > 0
        
    def test_calculate_measures_percentiles(self, mock_evaluator):
        """Test percentile calculations."""
        train_data = pd.DataFrame({
            'data': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        
        test_data = pd.DataFrame({
            'data': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        
        results = mock_evaluator.calculate_measures(
            train_data=train_data,
            test_data=test_data,
            feature_names=['data']
        )
        
        assert np.isclose(results['data']['train']['25_percentile'], 25.0)
        assert np.isclose(results['data']['train']['75_percentile'], 75.0)


class TestCategoricalStatistics:
    @pytest.fixture
    def mock_evaluator(self):
        evaluator = dataset_measures.CategoricalStatistics(
            method_name="test_categorical_stats",
            description="test description"
        )
        return evaluator
    
    @pytest.fixture
    def sample_categorical_data(self):
        train_data = pd.DataFrame({
            'category1': ['A', 'A', 'B', 'B', 'C'],
            'category2': ['X', 'Y', 'X', 'Y', 'Z'],
            'category3': ['red', 'blue', 'red', 'blue', 'green']
        })
        
        test_data = pd.DataFrame({
            'category1': ['A', 'B', 'B', 'C', 'C'],
            'category2': ['X', 'X', 'Y', 'Y', 'Z'],
            'category3': ['red', 'red', 'blue', 'blue', 'green']
        })
        
        return train_data, test_data
    
    def test_calculate_measures_single_feature(self, mock_evaluator, sample_categorical_data):
        """Test calculate_measures with a single feature."""
        train_data, test_data = sample_categorical_data
        
        results = mock_evaluator.calculate_measures(
            train_data=train_data,
            test_data=test_data,
            feature_names=['category1']
        )
        
        # Verify structure
        assert isinstance(results, dict)
        assert 'category1' in results
        assert 'train' in results['category1']
        assert 'test' in results['category1']
        assert 'chi_square' in results['category1']
    
    def test_calculate_measures_frequency_counts(self, mock_evaluator, sample_categorical_data):
        """Test frequency count calculations."""
        train_data, test_data = sample_categorical_data
        
        results = mock_evaluator.calculate_measures(
            train_data=train_data,
            test_data=test_data,
            feature_names=['category1']
        )
        
        train_freq = results['category1']['train']['frequency']
        assert train_freq['A'] == 2
        assert train_freq['B'] == 2
        assert train_freq['C'] == 1
        
        test_freq = results['category1']['test']['frequency']
        assert test_freq['A'] == 1
        assert test_freq['B'] == 2
        assert test_freq['C'] == 2
    
    def test_calculate_measures_proportions(self, mock_evaluator, sample_categorical_data):
        """Test proportion calculations."""
        train_data, test_data = sample_categorical_data
        
        results = mock_evaluator.calculate_measures(
            train_data=train_data,
            test_data=test_data,
            feature_names=['category1']
        )
        
        train_prop = results['category1']['train']['proportion']
        assert np.isclose(train_prop['A'], 2/5)
        assert np.isclose(train_prop['B'], 2/5)
        assert np.isclose(train_prop['C'], 1/5)
        
        assert np.isclose(sum(train_prop.values()), 1.0)
    
    def test_calculate_measures_num_unique(self, mock_evaluator):
        """Test number of unique values calculation."""
        train_data = pd.DataFrame({
            'few_categories': ['A', 'A', 'B', 'B', 'C'],
            'many_categories': ['A', 'B', 'C', 'D', 'E']
        })
        
        test_data = pd.DataFrame({
            'few_categories': ['A', 'A', 'A', 'B', 'B'],
            'many_categories': ['A', 'B', 'C', 'D', 'E']
        })
        
        results = mock_evaluator.calculate_measures(
            train_data=train_data,
            test_data=test_data,
            feature_names=['few_categories', 'many_categories']
        )
        
        assert results['few_categories']['train']['num_unique'] == 3
        assert results['few_categories']['test']['num_unique'] == 2
        assert results['many_categories']['train']['num_unique'] == 5
        assert results['many_categories']['test']['num_unique'] == 5
    
    def test_calculate_measures_entropy(self, mock_evaluator):
        """Test entropy calculation."""
        train_uniform = pd.DataFrame({
            'uniform': ['A', 'B', 'C', 'D']
        })
        
        train_skewed = pd.DataFrame({
            'skewed': ['A', 'A', 'A', 'B']
        })
        
        test_data = pd.DataFrame({
            'uniform': ['A', 'B', 'C', 'D'],
            'skewed': ['A', 'A', 'A', 'B']
        })
        
        train_data = pd.concat([train_uniform, train_skewed], axis=1)
        
        results = mock_evaluator.calculate_measures(
            train_data=train_data,
            test_data=test_data,
            feature_names=['uniform', 'skewed']
        )
        
        uniform_entropy = results['uniform']['train']['entropy']
        skewed_entropy = results['skewed']['train']['entropy']
        
        assert uniform_entropy > skewed_entropy
        assert np.isclose(uniform_entropy, 2.0)
    
    def test_calculate_measures_multiple_features(self, mock_evaluator, sample_categorical_data):
        """Test calculate_measures with multiple features."""
        train_data, test_data = sample_categorical_data
        
        results = mock_evaluator.calculate_measures(
            train_data=train_data,
            test_data=test_data,
            feature_names=['category1', 'category2', 'category3']
        )
        
        for feature in ['category1', 'category2', 'category3']:
            assert 'train' in results[feature]
            assert 'test' in results[feature]
            assert 'chi_square' in results[feature]
    
    def test_calculate_measures_chi_square_test(self, mock_evaluator):
        """Test chi-square test for distribution differences."""
        train_same = pd.DataFrame({
            'same_dist': ['A', 'A', 'B', 'B', 'C', 'C']
        })
        
        test_same = pd.DataFrame({
            'same_dist': ['A', 'A', 'B', 'B', 'C', 'C']
        })
        
        train_diff = pd.DataFrame({
            'diff_dist': ['A', 'A', 'A', 'B', 'C', 'C']
        })
        
        test_diff = pd.DataFrame({
            'diff_dist': ['A', 'B', 'B', 'B', 'C', 'C']
        })
        
        train_data = pd.concat([train_same, train_diff], axis=1)
        test_data = pd.concat([test_same, test_diff], axis=1)
        
        results = mock_evaluator.calculate_measures(
            train_data=train_data,
            test_data=test_data,
            feature_names=['same_dist', 'diff_dist']
        )
        
        same_p_value = results['same_dist']['chi_square']['p_value']
        assert same_p_value > 0.05
        assert results['same_dist']['chi_square']['chi2_stat'] >= 0
        assert results['diff_dist']['chi_square']['chi2_stat'] >= 0
        assert results['same_dist']['chi_square']['degrees_of_freedom'] == 2
    
    def test_calculate_measures_empty_feature_list(self, mock_evaluator, sample_categorical_data):
        """Test calculate_measures with empty feature list."""
        train_data, test_data = sample_categorical_data
        
        results = mock_evaluator.calculate_measures(
            train_data=train_data,
            test_data=test_data,
            feature_names=[]
        )
        
        assert isinstance(results, dict)
        assert len(results) == 0
    
    def test_calculate_measures_single_category(self, mock_evaluator):
        """Test with feature containing single category."""
        train_data = pd.DataFrame({
            'single_cat': ['A', 'A', 'A', 'A', 'A']
        })
        
        test_data = pd.DataFrame({
            'single_cat': ['A', 'A', 'A', 'A', 'A']
        })
        
        results = mock_evaluator.calculate_measures(
            train_data=train_data,
            test_data=test_data,
            feature_names=['single_cat']
        )
        
        assert results['single_cat']['train']['num_unique'] == 1
        assert results['single_cat']['train']['frequency']['A'] == 5
        assert np.isclose(results['single_cat']['train']['proportion']['A'], 1.0)
        assert np.isclose(results['single_cat']['train']['entropy'], 0.0)
    
    def test_calculate_measures_different_categories_train_test(self, mock_evaluator):
        """Test when train and test have different categories."""
        train_data = pd.DataFrame({
            'mixed': ['A', 'A', 'B', 'B', 'C']
        })
        
        test_data = pd.DataFrame({
            'mixed': ['B', 'C', 'C', 'D', 'D']
        })
        
        results = mock_evaluator.calculate_measures(
            train_data=train_data,
            test_data=test_data,
            feature_names=['mixed']
        )
        
        train_freq = results['mixed']['train']['frequency']
        assert 'A' in train_freq
        assert 'B' in train_freq
        assert 'C' in train_freq
        assert 'D' not in train_freq
        
        test_freq = results['mixed']['test']['frequency']
        assert 'A' not in test_freq
        assert 'B' in test_freq
        assert 'C' in test_freq
        assert 'D' in test_freq
        
        chi_square = results['mixed']['chi_square']
        assert np.isfinite(chi_square['chi2_stat'])
        assert np.isfinite(chi_square['p_value'])

