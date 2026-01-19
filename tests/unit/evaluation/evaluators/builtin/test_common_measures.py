"""Unit tests for common measure evaluators."""
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from brisk.evaluation.evaluators.builtin import common_measures

class TestEvaluateModel:
    """Tests for EvaluateModel class."""
    @pytest.fixture
    def mock_evaluator(self):
        evaluator = common_measures.EvaluateModel(
            method_name="test_evaluate",
            description="test description"
        )
        
        evaluator.metric_config = mock.Mock()
        evaluator.services = mock.Mock()
        
        return evaluator
    
    @pytest.fixture
    def sample_data(self):
        y_true = pd.Series([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        predictions = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 1, 1])
        return predictions, y_true
    
    def test_calculate_measures_no_metric(self, mock_evaluator, sample_data):
        """Test calculate_measures with no metrics returns empty dict."""
        predictions, y_true = sample_data
        
        results = mock_evaluator.calculate_measures(
            predictions=predictions,
            y_true=y_true,
            metrics=[]
        )
        
        assert results == {}
    
    def test_calculate_measures_one_metric(self, mock_evaluator, sample_data):
        """Test calculate_measures with one metric."""
        predictions, y_true = sample_data
        
        mock_scorer = mock.Mock(return_value=0.85)
        mock_evaluator.metric_config.get_metric.return_value = mock_scorer
        mock_evaluator.metric_config.get_name.return_value = "Accuracy"
        
        results = mock_evaluator.calculate_measures(
            predictions=predictions,
            y_true=y_true,
            metrics=["accuracy"]
        )
        
        assert results["Accuracy"] == 0.85
        mock_scorer.assert_called_once_with(y_true, predictions)
    
    def test_calculate_measures_two_metrics(self, mock_evaluator, sample_data):
        """Test calculate_measures with two metrics."""
        predictions, y_true = sample_data
        
        def mock_get_metric(metric_name):
            if metric_name == "accuracy":
                return mock.Mock(return_value=0.80)
            elif metric_name == "f1_score":
                return mock.Mock(return_value=0.75)
            return None
        
        def mock_get_name(metric_name):
            if metric_name == "accuracy":
                return "Accuracy"
            elif metric_name == "f1_score":
                return "F1 Score"
            return metric_name
        
        mock_evaluator.metric_config.get_metric.side_effect = mock_get_metric
        mock_evaluator.metric_config.get_name.side_effect = mock_get_name
        
        results = mock_evaluator.calculate_measures(
            predictions=predictions,
            y_true=y_true,
            metrics=["accuracy", "f1_score"]
        )
        
        assert len(results) == 2
        assert results["Accuracy"] == 0.80
        assert results["F1 Score"] == 0.75
    
    def test_calculate_measures_missing_scorer(self, mock_evaluator, sample_data):
        """Test calculate_measures with missing scorer logs warning and skips."""
        predictions, y_true = sample_data
        
        mock_evaluator.metric_config.get_metric.return_value = None
        mock_evaluator.metric_config.get_name.return_value = "Missing Metric"
        
        results = mock_evaluator.calculate_measures(
            predictions=predictions,
            y_true=y_true,
            metrics=["missing_metric"]
        )
        
        assert results == {}
        
        mock_evaluator.services.logger.logger.info.assert_called_once()
        log_call = mock_evaluator.services.logger.logger.info.call_args[0][0]
        assert "missing_metric" in log_call


class TestEvaluateModelCV:
    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock EvaluateModelCV evaluator with configured dependencies."""
        evaluator = common_measures.EvaluateModelCV(
            method_name="test_evaluate_cv",
            description="test description"
        )
        
        evaluator.metric_config = mock.Mock()
        evaluator.services = mock.Mock()
        
        return evaluator
    
    @pytest.fixture
    def sample_cv_data(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3), columns=['f1', 'f2', 'f3'])
        y = pd.Series(np.random.randint(0, 2, 50))
        return X, y
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = mock.Mock()
        model.wrapper_name = "test_wrapper"
        return model
    
    def test_calculate_measures_no_metric(self, mock_evaluator, mock_model, sample_cv_data):
        X, y = sample_cv_data
        
        mock_splitter = mock.Mock()
        mock_evaluator.utility.get_cv_splitter.return_value = (mock_splitter, None)
        
        results = mock_evaluator.calculate_measures(
            model=mock_model,
            X=X,
            y=y,
            metrics=[],
            cv=5
        )
        
        assert results == {}
    
    @mock.patch('sklearn.model_selection.cross_val_score')
    def test_calculate_measures_one_metric(
        self, mock_cross_val_score, mock_evaluator, mock_model, sample_cv_data
    ):
        """Test calculate_measures with one metric."""
        X, y = sample_cv_data
        
        # Mock cross_val_score to return deterministic scores
        mock_scores = np.array([0.80, 0.85, 0.75, 0.90, 0.82])
        mock_cross_val_score.return_value = mock_scores
        
        mock_splitter = mock.Mock()
        mock_evaluator.utility.get_cv_splitter.return_value = (mock_splitter, None)
        mock_scorer = mock.Mock()
        mock_evaluator.metric_config.get_scorer.return_value = mock_scorer
        mock_evaluator.metric_config.get_name.return_value = "Accuracy"
        
        results = mock_evaluator.calculate_measures(
            model=mock_model,
            X=X,
            y=y,
            metrics=["accuracy"],
            cv=5
        )
        
        assert np.isclose(results["Accuracy"]["mean_score"], mock_scores.mean())
        assert np.isclose(results["Accuracy"]["std_dev"], mock_scores.std())
        assert results["Accuracy"]["all_scores"] == mock_scores.tolist()
    
    @mock.patch('sklearn.model_selection.cross_val_score')
    def test_calculate_measures_two_metrics(
        self, mock_cross_val_score, mock_evaluator, mock_model, sample_cv_data
    ):
        """Test calculate_measures with two metrics."""
        X, y = sample_cv_data
        
        # Mock cross_val_score to return different scores for each metric
        accuracy_scores = np.array([0.80, 0.85, 0.75, 0.90, 0.82])
        f1_scores = np.array([0.70, 0.75, 0.65, 0.80, 0.72])
        
        mock_cross_val_score.side_effect = [accuracy_scores, f1_scores]
        
        mock_splitter = mock.Mock()
        mock_evaluator.utility.get_cv_splitter.return_value = (mock_splitter, None)
        
        def mock_get_scorer(metric_name):
            return mock.Mock(name=f"scorer_{metric_name}")
        
        def mock_get_name(metric_name):
            if metric_name == "accuracy":
                return "Accuracy"
            elif metric_name == "f1_score":
                return "F1 Score"
            return metric_name
        
        mock_evaluator.metric_config.get_scorer.side_effect = mock_get_scorer
        mock_evaluator.metric_config.get_name.side_effect = mock_get_name
        
        results = mock_evaluator.calculate_measures(
            model=mock_model,
            X=X,
            y=y,
            metrics=["accuracy", "f1_score"],
            cv=5
        )
        
        # Verify Accuracy statistics
        assert np.isclose(results["Accuracy"]["mean_score"], accuracy_scores.mean())
        assert np.isclose(results["Accuracy"]["std_dev"], accuracy_scores.std())
        assert results["Accuracy"]["all_scores"] == accuracy_scores.tolist()
        
        # Verify F1 Score statistics
        assert np.isclose(results["F1 Score"]["mean_score"], f1_scores.mean())
        assert np.isclose(results["F1 Score"]["std_dev"], f1_scores.std())
        assert results["F1 Score"]["all_scores"] == f1_scores.tolist()
    
    def test_calculate_measures_missing_scorer(
        self, mock_evaluator, mock_model, sample_cv_data
    ):
        """Test calculate_measures with missing scorer logs warning and skips."""
        X, y = sample_cv_data
        
        mock_splitter = mock.Mock()
        mock_evaluator.utility.get_cv_splitter.return_value = (mock_splitter, None)
        mock_evaluator.metric_config.get_scorer.return_value = None
        mock_evaluator.metric_config.get_name.return_value = "Missing Metric"
        
        results = mock_evaluator.calculate_measures(
            model=mock_model,
            X=X,
            y=y,
            metrics=["missing_metric"],
            cv=5
        )
        
        assert results == {}
        
        mock_evaluator.services.logger.logger.info.assert_called_once()
        log_call = mock_evaluator.services.logger.logger.info.call_args[0][0]
        assert "missing_metric" in log_call
    
    @mock.patch('sklearn.model_selection.cross_val_score')
    def test_calculate_measures_cv_0(
        self, mock_cross_val_score, mock_evaluator, mock_model, sample_cv_data
    ):
        """Test calculate_measures with cv=0 (edge case)."""
        X, y = sample_cv_data
        
        mock_cross_val_score.return_value = np.array([])
        
        mock_splitter = mock.Mock()
        mock_evaluator.utility.get_cv_splitter.return_value = (mock_splitter, None)
        mock_scorer = mock.Mock()
        mock_evaluator.metric_config.get_scorer.return_value = mock_scorer
        mock_evaluator.metric_config.get_name.return_value = "Accuracy"
        
        results = mock_evaluator.calculate_measures(
            model=mock_model,
            X=X,
            y=y,
            metrics=["accuracy"],
            cv=0
        )
        
        assert np.isnan(results["Accuracy"]["mean_score"])
        assert np.isnan(results["Accuracy"]["std_dev"])
        assert results["Accuracy"]["all_scores"] == []


class TestCompareModels:
    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock CompareModels evaluator with configured dependencies."""
        evaluator = common_measures.CompareModels(
            method_name="test_compare",
            description="test description"
        )
        
        evaluator.metric_config = mock.Mock()
        evaluator.services = mock.Mock()
        # evaluator.utility = Mock()
        
        return evaluator
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for model comparison."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3), columns=['f1', 'f2', 'f3'])
        y = pd.Series(np.random.randint(0, 2, 50))
        return X, y
    
    @pytest.fixture
    def mock_models(self):
        """Create mock models for testing."""
        model1 = mock.Mock()
        model1.wrapper_name = "model1_wrapper"
        model1.predict = mock.Mock(return_value=np.array([0, 1, 0, 1, 1]))
        
        model2 = mock.Mock()
        model2.wrapper_name = "model2_wrapper"
        model2.predict = mock.Mock(return_value=np.array([1, 1, 0, 0, 1]))
        
        return model1, model2
    
    def test_calculate_measures_no_metric(self, mock_evaluator, mock_models, sample_data):
        """Test calculate_measures with no metrics returns model names only."""
        X, y = sample_data
        model1, model2 = mock_models
        
        def mock_get_wrapper(wrapper_name):
            wrapper = mock.Mock()
            if wrapper_name == "model1_wrapper":
                wrapper.display_name = "Model 1"
            elif wrapper_name == "model2_wrapper":
                wrapper.display_name = "Model 2"
            return wrapper
        
        mock_evaluator.utility.get_algo_wrapper.side_effect = mock_get_wrapper
        
        results = mock_evaluator.calculate_measures(
            model1, model2,
            X=X,
            y=y,
            metrics=[],
            calculate_diff=False
        )
        
        assert results["Model 1"] == {}
        assert results["Model 2"] == {}
    
    def test_calculate_measures_one_metric(self, mock_evaluator, mock_models, sample_data):
        """Test calculate_measures with one metric."""
        X, y = sample_data
        model1, model2 = mock_models
        
        def mock_get_wrapper(wrapper_name):
            wrapper = mock.Mock()
            if wrapper_name == "model1_wrapper":
                wrapper.display_name = "Model 1"
            elif wrapper_name == "model2_wrapper":
                wrapper.display_name = "Model 2"
            return wrapper
        
        mock_evaluator.utility.get_algo_wrapper.side_effect = mock_get_wrapper
        
        mock_scorer = mock.Mock(side_effect=[0.85, 0.78])
        mock_evaluator.metric_config.get_metric.return_value = mock_scorer
        mock_evaluator.metric_config.get_name.return_value = "Accuracy"
        
        results = mock_evaluator.calculate_measures(
            model1, model2,
            X=X,
            y=y,
            metrics=["accuracy"],
            calculate_diff=False
        )
        
        assert results["Model 1"]["Accuracy"] == 0.85
        assert results["Model 2"]["Accuracy"] == 0.78
        assert mock_scorer.call_count == 2
    
    def test_calculate_measures_two_metrics(self, mock_evaluator, mock_models, sample_data):
        """Test calculate_measures with two metrics."""
        X, y = sample_data
        model1, model2 = mock_models
        
        def mock_get_wrapper(wrapper_name):
            wrapper = mock.Mock()
            if wrapper_name == "model1_wrapper":
                wrapper.display_name = "Model 1"
            elif wrapper_name == "model2_wrapper":
                wrapper.display_name = "Model 2"
            return wrapper
        
        mock_evaluator.utility.get_algo_wrapper.side_effect = mock_get_wrapper
        
        accuracy_scorer = mock.Mock(side_effect=[0.85, 0.78])
        f1_scorer = mock.Mock(side_effect=[0.80, 0.75])
        
        def mock_get_metric(metric_name):
            if metric_name == "accuracy":
                return accuracy_scorer
            elif metric_name == "f1_score":
                return f1_scorer
            return None
        
        def mock_get_name(metric_name):
            if metric_name == "accuracy":
                return "Accuracy"
            elif metric_name == "f1_score":
                return "F1 Score"
            return metric_name
        
        mock_evaluator.metric_config.get_metric.side_effect = mock_get_metric
        mock_evaluator.metric_config.get_name.side_effect = mock_get_name
        
        results = mock_evaluator.calculate_measures(
            model1, model2,
            X=X,
            y=y,
            metrics=["accuracy", "f1_score"],
            calculate_diff=False
        )
        
        assert results["Model 1"]["Accuracy"] == 0.85
        assert results["Model 1"]["F1 Score"] == 0.80
        assert results["Model 2"]["Accuracy"] == 0.78
        assert results["Model 2"]["F1 Score"] == 0.75
    
    def test_calculate_measures_missing_scorer(
        self, mock_evaluator, mock_models, sample_data
    ):
        """Test calculate_measures with missing scorer logs warning and skips."""
        X, y = sample_data
        model1, _ = mock_models

        mock_wrapper = mock.Mock()
        mock_wrapper.display_name = "Model 1"
        mock_evaluator.utility.get_algo_wrapper.return_value = mock_wrapper
        mock_evaluator.metric_config.get_metric.return_value = None
        mock_evaluator.metric_config.get_name.return_value = "Missing Metric"
        
        results = mock_evaluator.calculate_measures(
            model1,
            X=X,
            y=y,
            metrics=["missing_metric"],
            calculate_diff=False
        )
        
        assert results["Model 1"] == {}
        
        mock_evaluator.services.logger.logger.info.assert_called_once()
        log_call = mock_evaluator.services.logger.logger.info.call_args[0][0]
        assert "missing_metric" in log_call
    
    def test_calculate_measures_one_model(self, mock_evaluator, mock_models, sample_data):
        """Test calculate_measures with one model works correctly."""
        X, y = sample_data
        model1, _ = mock_models
        
        mock_wrapper = mock.Mock()
        mock_wrapper.display_name = "Model 1"
        mock_evaluator.utility.get_algo_wrapper.return_value = mock_wrapper
        
        mock_scorer = mock.Mock(return_value=0.85)
        mock_evaluator.metric_config.get_metric.return_value = mock_scorer
        mock_evaluator.metric_config.get_name.return_value = "Accuracy"
        
        results = mock_evaluator.calculate_measures(
            model1,
            X=X,
            y=y,
            metrics=["accuracy"],
            calculate_diff=False
        )
        
        assert results["Model 1"]["Accuracy"] == 0.85
        assert "differences" not in results
    
    def test_calculate_measures_calculate_diff(
        self, mock_evaluator, mock_models, sample_data
    ):
        """Test calculate_measures with calculate_diff=True computes differences."""
        X, y = sample_data
        model1, model2 = mock_models
        
        def mock_get_wrapper(wrapper_name):
            wrapper = mock.Mock()
            if wrapper_name == "model1_wrapper":
                wrapper.display_name = "Model 1"
            elif wrapper_name == "model2_wrapper":
                wrapper.display_name = "Model 2"
            return wrapper
        
        mock_evaluator.utility.get_algo_wrapper.side_effect = mock_get_wrapper
        
        accuracy_scorer = mock.Mock(side_effect=[0.85, 0.78])
        f1_scorer = mock.Mock(side_effect=[0.80, 0.90])
        
        def mock_get_metric(metric_name):
            if metric_name == "accuracy":
                return accuracy_scorer
            elif metric_name == "f1_score":
                return f1_scorer
            return None
        
        def mock_get_name(metric_name):
            if metric_name == "accuracy":
                return "Accuracy"
            elif metric_name == "f1_score":
                return "F1 Score"
            return metric_name
        
        mock_evaluator.metric_config.get_metric.side_effect = mock_get_metric
        mock_evaluator.metric_config.get_name.side_effect = mock_get_name
        
        results = mock_evaluator.calculate_measures(
            model1, model2,
            X=X,
            y=y,
            metrics=["accuracy", "f1_score"],
            calculate_diff=True
        )
        
        assert isinstance(results["differences"], dict)
        
        print(results)
        accuracy_diff = results["differences"]["Accuracy"]["Model 2 - Model 1"]
        print(accuracy_diff)
        assert np.isclose(accuracy_diff, 0.78 - 0.85)
        
        f1_diff = results["differences"]["F1 Score"]["Model 2 - Model 1"]
        assert np.isclose(f1_diff, 0.90 - 0.80)
