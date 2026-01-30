"""Unit tests for common plots."""
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn import tree, ensemble, linear_model

from brisk.evaluation.evaluators.builtin import common_plots
from brisk.theme import plot_settings


@pytest.mark.unit
class TestPlotLearningCurve:
    @pytest.fixture
    def mock_evaluator(self):
        evaluator = common_plots.PlotLearningCurve(
            method_name="test_learning_curve",
            description="test description",
            plot_settings=plot_settings.PlotSettings()
        )

        evaluator.metric_config = mock.Mock()
        evaluator.services = mock.Mock()

        return evaluator

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
        y = pd.Series(np.random.randn(100))
        return X, y

    @pytest.fixture
    def mock_model(self):
        model = mock.Mock()
        model.wrapper_name = "test_wrapper"
        return model

    @mock.patch('sklearn.model_selection.learning_curve')
    def test_generate_plot_data_0_cv(
        self, mock_learning_curve, mock_evaluator, mock_model, sample_data
    ):
        """Test generate_plot_data with cv=0 (edge case)."""
        X, y = sample_data

        mock_learning_curve.return_value = (
            np.array([]),  # train_sizes
            np.array([]).reshape(0, 0),
            np.array([]).reshape(0, 0),
            np.array([]).reshape(0, 0),
            np.array([]).reshape(0, 0)
        )

        mock_splitter = mock.Mock()
        mock_evaluator.utility.get_cv_splitter.return_value = (mock_splitter, None)
        mock_scorer = mock.Mock()
        mock_evaluator.metric_config.get_scorer.return_value = mock_scorer

        results = mock_evaluator.generate_plot_data(
            model=mock_model,
            X=X,
            y=y,
            cv=0,
            num_repeats=1,
            n_jobs=-1,
            metric="neg_mean_absolute_error"
        )

        assert len(results["train_sizes"]) == 0
        assert len(results["train_scores_mean"]) == 0
        assert len(results["test_scores_mean"]) == 0

    @mock.patch('sklearn.model_selection.learning_curve')
    def test_generate_plot_data(
        self, mock_learning_curve, mock_evaluator, mock_model, sample_data
    ):
        """Test generate_plot_data with normal cv value."""
        X, y = sample_data

        train_sizes = np.array([10, 25, 50, 75, 100])
        train_scores = np.random.rand(5, 5) * 0.2 + 0.8
        test_scores = np.random.rand(5, 5) * 0.3 + 0.6
        fit_times = np.random.rand(5, 5) * 2 + 0.5
        score_times = np.random.rand(5, 5) * 0.1

        mock_learning_curve.return_value = (
            train_sizes, train_scores, test_scores, fit_times, score_times
        )

        mock_splitter = mock.Mock()
        mock_evaluator.utility.get_cv_splitter.return_value = (mock_splitter, None)
        mock_scorer = mock.Mock()
        mock_evaluator.metric_config.get_scorer.return_value = mock_scorer

        results = mock_evaluator.generate_plot_data(
            model=mock_model,
            X=X,
            y=y,
            cv=5,
            num_repeats=1,
            n_jobs=-1,
            metric="neg_mean_absolute_error"
        )

        np.testing.assert_array_equal(results["train_sizes"], train_sizes)

        # Verify means and stds are calculated correctly
        np.testing.assert_array_almost_equal(
            results["train_scores_mean"], np.mean(train_scores, axis=1)
        )
        np.testing.assert_array_almost_equal(
            results["train_scores_std"], np.std(train_scores, axis=1)
        )
        np.testing.assert_array_almost_equal(
            results["test_scores_mean"], np.mean(test_scores, axis=1)
        )
        np.testing.assert_array_almost_equal(
            results["test_scores_std"], np.std(test_scores, axis=1)
        )
        np.testing.assert_array_almost_equal(
            results["fit_times_mean"], np.mean(fit_times, axis=1)
        )
        np.testing.assert_array_almost_equal(
            results["fit_times_std"], np.std(fit_times, axis=1)
        )


@pytest.mark.unit
class TestPlotFeatureImportance:
    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock PlotFeatureImportance evaluator."""
        evaluator = common_plots.PlotFeatureImportance(
            method_name="test_feature_importance",
            description="test description",
            plot_settings=plot_settings.PlotSettings()
        )

        evaluator.metric_config = mock.Mock()
        evaluator.services = mock.Mock()

        return evaluator

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(50, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        y = pd.Series(np.random.randn(50))
        feature_names = X.columns.tolist()
        return X, y, feature_names

    def test_generate_plot_data_decision_tree_detected(
        self, mock_evaluator, sample_data
    ):
        """Test that DecisionTreeRegressor uses built-in feature_importances_."""
        X, y, feature_names = sample_data

        model = tree.DecisionTreeRegressor(random_state=42, max_depth=3)

        mock_evaluator.metric_config.get_scorer.return_value = mock.Mock()

        importance_data, width, height = mock_evaluator.generate_plot_data(
            model=model,
            X=X,
            y=y,
            threshold=5,
            feature_names=feature_names,
            metric="accuracy",
            num_rep=10
        )

        assert hasattr(model, 'tree_')
        assert len(importance_data) == 5
        assert isinstance(importance_data["Feature"].dtype, pd.CategoricalDtype)

    def test_generate_plot_data_random_forest_detected(
        self, mock_evaluator, sample_data
    ):
        """Test that RandomForestRegressor uses built-in feature_importances_."""
        X, y, feature_names = sample_data

        model = ensemble.RandomForestRegressor(n_estimators=10, random_state=42, max_depth=3)

        mock_evaluator.metric_config.get_scorer.return_value = mock.Mock()

        importance_data, width, height = mock_evaluator.generate_plot_data(
            model=model,
            X=X,
            y=y,
            threshold=5,
            feature_names=feature_names,
            metric="accuracy",
            num_rep=10
        )

        assert hasattr(model, 'estimators_')
        assert len(importance_data) == 5

    def test_generate_plot_data_gradient_boosting_detected(
        self, mock_evaluator, sample_data
    ):
        """Test that GradientBoostingRegressor uses built-in feature_importances_."""
        X, y, feature_names = sample_data

        model = ensemble.GradientBoostingRegressor(
            n_estimators=10, random_state=42, max_depth=3
        )

        mock_evaluator.metric_config.get_scorer.return_value = mock.Mock()

        importance_data, width, height = mock_evaluator.generate_plot_data(
            model=model,
            X=X,
            y=y,
            threshold=5,
            feature_names=feature_names,
            metric="accuracy",
            num_rep=10
        )

        assert hasattr(model, 'estimators_')
        assert len(importance_data) == 5

    @mock.patch('sklearn.inspection.permutation_importance')
    def test_generate_plot_data_permutation_importance(
        self, mock_perm_importance, mock_evaluator, sample_data
    ):
        """Test that non-tree models use permutation importance."""
        X, y, feature_names = sample_data

        model = linear_model.LinearRegression()

        mock_result = mock.Mock()
        mock_result.importances_mean = np.random.rand(10)
        mock_perm_importance.return_value = mock_result

        mock_scorer = mock.Mock()
        mock_evaluator.metric_config.get_scorer.return_value = mock_scorer

        importance_data, width, height = mock_evaluator.generate_plot_data(
            model=model,
            X=X,
            y=y,
            threshold=5,
            feature_names=feature_names,
            metric="r2",
            num_rep=10
        )

        mock_perm_importance.assert_called_once()
        assert hasattr(model, 'coef_')
        assert len(importance_data) == 5

    def test_generate_plot_data_int_threshold(self, mock_evaluator, sample_data):
        """Test that integer threshold returns top N features."""
        X, y, feature_names = sample_data

        model = tree.DecisionTreeRegressor(random_state=42, max_depth=3)
        mock_evaluator.metric_config.get_scorer.return_value = mock.Mock()

        importance_data, width, height = mock_evaluator.generate_plot_data(
            model=model,
            X=X,
            y=y,
            threshold=3,
            feature_names=feature_names,
            metric="accuracy",
            num_rep=10
        )

        assert len(importance_data) == 3
        importances = importance_data["Importance"].values
        assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))

    def test_generate_plot_data_float_threshold(self, mock_evaluator, sample_data):
        """Test that float threshold returns proportion of features."""
        X, y, feature_names = sample_data

        model = tree.DecisionTreeRegressor(random_state=42, max_depth=3)
        mock_evaluator.metric_config.get_scorer.return_value = mock.Mock()

        importance_data, width, height = mock_evaluator.generate_plot_data(
            model=model,
            X=X,
            y=y,
            threshold=0.3,
            feature_names=feature_names,
            metric="accuracy",
            num_rep=10
        )

        assert len(importance_data) == 3

    def test_generate_plot_data_float_threshold_round_up(self, mock_evaluator, sample_data):
        """Test that float threshold rounds up."""
        X, y, feature_names = sample_data

        model = tree.DecisionTreeRegressor(random_state=42, max_depth=3)
        mock_evaluator.metric_config.get_scorer.return_value = mock.Mock()
        importance_data, width, height = mock_evaluator.generate_plot_data(
            model=model,
            X=X,
            y=y,
            threshold=0.05,
            feature_names=feature_names,
            metric="accuracy",
            num_rep=10
        )

        assert len(importance_data) == 1


@pytest.mark.unit
class TestPlotModelComparison:
    @pytest.fixture
    def mock_evaluator(self):
        evaluator = common_plots.PlotModelComparison(
            method_name="test_model_comparison",
            description="test description",
            plot_settings=plot_settings.PlotSettings()
        )

        evaluator.metric_config = mock.Mock()
        evaluator.services = mock.Mock()

        return evaluator

    @pytest.fixture
    def sample_data(self):
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

    def test_generate_plot_data_no_models(self, mock_evaluator, sample_data):
        """Test generate_plot_data with no models returns empty data."""
        X, y = sample_data

        mock_scorer = mock.Mock(return_value=0.85)
        mock_evaluator.metric_config.get_metric.return_value = mock_scorer

        plot_data = mock_evaluator.generate_plot_data(
            X=X,
            y=y,
            metric="accuracy"
        )

        assert len(plot_data) == 0
        assert list(plot_data.columns) == ["Model", "Score"]

    def test_generate_plot_data_one_model(
        self, mock_evaluator, mock_models, sample_data
    ):
        """Test generate_plot_data with one model."""
        X, y = sample_data
        model1, _ = mock_models

        mock_wrapper = mock.Mock()
        mock_wrapper.display_name = "Model 1"
        mock_evaluator.utility.get_algo_wrapper.return_value = mock_wrapper

        mock_scorer = mock.Mock(return_value=0.857142)
        mock_evaluator.metric_config.get_metric.return_value = mock_scorer

        plot_data = mock_evaluator.generate_plot_data(
            model1,
            X=X,
            y=y,
            metric="accuracy"
        )

        assert len(plot_data) == 1
        assert plot_data.iloc[0]["Model"] == "Model 1"
        assert plot_data.iloc[0]["Score"] == 0.857

    def test_generate_plot_data_two_models(
        self, mock_evaluator, mock_models, sample_data
    ):
        """Test generate_plot_data with two models."""
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

        mock_scorer = mock.Mock(side_effect=[0.857142, 0.714285])
        mock_evaluator.metric_config.get_metric.return_value = mock_scorer

        plot_data = mock_evaluator.generate_plot_data(
            model1, model2,
            X=X,
            y=y,
            metric="accuracy"
        )

        assert len(plot_data) == 2
        assert plot_data.iloc[0]["Score"] == 0.857
        assert plot_data.iloc[1]["Score"] == 0.714


@pytest.mark.unit
class TestPlotShapleyValues:
    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock PlotShapleyValues evaluator."""
        evaluator = common_plots.PlotShapleyValues(
            method_name="test_shapley",
            description="test description",
            plot_settings=plot_settings.PlotSettings()
        )

        evaluator.metric_config = mock.Mock()
        evaluator.services = mock.Mock()
        return evaluator

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(50, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        y = pd.Series(np.random.randn(50))
        return X, y

    @mock.patch('shap.TreeExplainer')
    def test_generate_plot_data_hasattr_tree(
        self, mock_tree_explainer, mock_evaluator, sample_data
    ):
        """Test that models with tree_ attribute use TreeExplainer."""
        X, y = sample_data

        model = mock.Mock()
        model.tree_ = mock.Mock()
        mock_explainer_instance = mock.Mock()
        mock_shap_values = mock.Mock()
        mock_shap_values.values = np.random.randn(50, 5)
        mock_explainer_instance.return_value = mock_shap_values
        mock_tree_explainer.return_value = mock_explainer_instance

        plot_data = mock_evaluator.generate_plot_data(
            model=model,
            X=X,
            y=y
        )

        mock_tree_explainer.assert_called_once_with(model)
        assert plot_data["X_sample"] is X

    @mock.patch('shap.TreeExplainer')
    def test_generate_plot_data_hasattr_estimators(
        self, mock_tree_explainer, mock_evaluator, sample_data
    ):
        """Test that models with estimators_ attribute use TreeExplainer."""
        X, y = sample_data

        model = mock.Mock()
        model.estimators_ = [mock.Mock(), mock.Mock()]

        mock_explainer_instance = mock.Mock()
        mock_shap_values = mock.Mock()
        mock_shap_values.values = np.random.randn(50, 5)
        mock_explainer_instance.return_value = mock_shap_values
        mock_tree_explainer.return_value = mock_explainer_instance

        plot_data = mock_evaluator.generate_plot_data(
            model=model,
            X=X,
            y=y
        )

        mock_tree_explainer.assert_called_once_with(model)

        assert isinstance(plot_data, dict)

    @mock.patch('shap.LinearExplainer')
    def test_generate_plot_data_hasattr_coef(
        self, mock_linear_explainer, mock_evaluator, sample_data
    ):
        """Test that models with coef_ attribute use LinearExplainer."""
        X, y = sample_data

        model = mock.Mock()
        model.coef_ = np.array([0.5, -0.3, 0.8, 0.1, -0.2])
        delattr(model, 'tree_') if hasattr(model, 'tree_') else None
        delattr(model, 'estimators_') if hasattr(model, 'estimators_') else None

        mock_explainer_instance = mock.Mock()
        mock_shap_values = mock.Mock()
        mock_shap_values.values = np.random.randn(50, 5)
        mock_explainer_instance.return_value = mock_shap_values
        mock_linear_explainer.return_value = mock_explainer_instance

        plot_data = mock_evaluator.generate_plot_data(
            model=model,
            X=X,
            y=y
        )

        mock_linear_explainer.assert_called_once_with(model, X)

        assert "shap_values" in plot_data
        assert "X_sample" in plot_data

    @mock.patch('shap.KernelExplainer')
    @mock.patch('shap.sample')
    def test_generate_plot_data_kernel_explainer(
        self, mock_shap_sample, mock_kernel_explainer, mock_evaluator, sample_data
    ):
        """Test that models without tree_/estimators_/coef_ use KernelExplainer."""
        X, y = sample_data

        model = mock.Mock()
        model.predict = mock.Mock(return_value=np.random.randn(50))
        if hasattr(model, 'tree_'):
            delattr(model, 'tree_')
        if hasattr(model, 'estimators_'):
            delattr(model, 'estimators_')
        if hasattr(model, 'coef_'):
            delattr(model, 'coef_')

        mock_background = X.sample(50, random_state=42)
        mock_shap_sample.return_value = mock_background

        # Mock the KernelExplainer
        mock_explainer_instance = mock.Mock()
        mock_shap_values = mock.Mock()
        mock_shap_values.values = np.random.randn(50, 5)
        mock_explainer_instance.return_value = mock_shap_values
        mock_kernel_explainer.return_value = mock_explainer_instance

        plot_data = mock_evaluator.generate_plot_data(
            model=model,
            X=X,
            y=y
        )

        mock_shap_sample.assert_called_once()
        call_args = mock_shap_sample.call_args[0]
        assert call_args[0] is X
        assert call_args[1] == min(100, len(X))

        # Verify KernelExplainer was used
        mock_kernel_explainer.assert_called_once_with(
            model.predict, mock_background
        )
