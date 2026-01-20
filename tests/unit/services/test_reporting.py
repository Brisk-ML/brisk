"""Unit tests for ReportingService."""

import pytest
import pandas as pd
from unittest import mock
from collections import namedtuple

from brisk.services import reporting
from brisk.reporting import report_data
from tests.utils import factories

@pytest.fixture
def mock_data_manager():
    manager = mock.MagicMock()
    manager.test_size = 0.2
    manager.n_splits = 3
    manager.split_method = "StratifiedKFold"
    manager.group_column = None
    manager.stratified = True
    manager.random_state = 42
    return manager


@pytest.fixture
def mock_utility_service():
    service = mock.MagicMock()
    
    def get_algo_wrapper(name):
        wrapper = mock.MagicMock()
        wrapper.display_name = f"{name} Display"
        return wrapper
    
    service.get_algo_wrapper = get_algo_wrapper
    return service


@pytest.fixture
def mock_logging_service():
    """Create a mock logging service."""
    service = mock.MagicMock()
    service.logger = mock.MagicMock()
    return service


@pytest.fixture
def mock_metric_manager():
    """Create a mock MetricManager."""
    manager = mock.MagicMock()
    manager._resolve_identifier = mock.MagicMock(return_value="accuracy")
    manager._metrics_by_name = {
        "accuracy": mock.MagicMock(abbr="Acc", display_name="Accuracy")
    }
    manager.is_higher_better = mock.MagicMock(return_value=True)
    return manager


@pytest.fixture
def mock_evaluator_registry():
    registry = mock.MagicMock()
    
    def get_evaluator(name):
        evaluator = mock.MagicMock()
        evaluator.method_name = name
        evaluator.description = f"{name} description"
        evaluator.report = mock.MagicMock(return_value=(["Column1", "Column2"], [["val1", "val2"]]))
        return evaluator
    
    registry.get = get_evaluator
    return registry


@pytest.fixture
def reporting_service(mock_utility_service, mock_logging_service, mock_metric_manager, mock_evaluator_registry):
    """Create a ReportingService instance with mocked dependencies."""
    service = reporting.ReportingService("test_reporting")
    service._other_services = {
        "utility": mock_utility_service,
        "logging": mock_logging_service
    }
    service.set_metric_config(mock_metric_manager)
    service.set_evaluator_registry(mock_evaluator_registry)
    return service


@pytest.fixture
def mock_algorithm():
    algorithm = {
        "model": mock.MagicMock(
            hyperparam_grid={"alpha": [0.1, 1.0, 10.0], "max_iter": [100, 200]}
        )
    }
    return algorithm


@pytest.fixture
def mock_experiment_group():
    Group = namedtuple("Group", ["name", "description", "datasets"])
    return Group(
        name="classification",
        description="Classification experiments",
        datasets=[("iris.csv", None), ("wine.csv", None)]
    )


class TestReportingService:
    def test_add_data_manager_clears_caches(self, reporting_service, mock_data_manager):
        """Test that add_data_manager clears internal caches."""
        reporting_service._image_cache[("g", "d", "s", "m")] = ("img", {})
        reporting_service._table_cache[("g", "d", "s", "m")] = ({}, {})
        reporting_service._cached_tuned_params = {"param": "value"}
        
        reporting_service.add_data_manager("test_group", mock_data_manager)
        
        assert reporting_service._image_cache == {}
        assert reporting_service._table_cache == {}
        assert reporting_service._cached_tuned_params == {}

    def test_add_dataset_categorical(self, reporting_service):
        """Test add_dataset with categorical target (uses proportion and entropy)."""
        splits = factories.DataSplitsFactory.simple(
            n_splits=2,
            group_name="classification",
            dataset_name="test_data.csv",
            problem_type="binary"
        )
        
        for split in splits._data_splits:
            split.y_train = pd.Series([0, 1] * 40, name="target")
            split.y_test = pd.Series([0, 1] * 10, name="target")
        
        mock_plot_data = report_data.PlotData(
            name="test_plot",
            description="test description",
            image="<svg>test</svg>"
        )
        
        with mock.patch.object(
            reporting_service, 
            "_create_plot_data", 
            return_value=mock_plot_data
        ):
            with mock.patch.object(
                reporting_service,
                "_create_feature_distribution",
                return_value=None
            ):
                reporting_service.add_dataset("classification", splits)
        
        dataset_id = "classification_test_data.csv"
        assert dataset_id in reporting_service.datasets
        
        dataset = reporting_service.datasets[dataset_id]
        
        for split_id, stats in dataset.split_target_stats.items():
            assert "proportion" in stats
            assert "entropy" in stats

    def test_add_dataset_continuous(self, reporting_service):
        """Test add_dataset with continuous target (uses mean, std, min, max)."""
        splits = factories.DataSplitsFactory.simple(
            n_splits=2,
            group_name="regression",
            dataset_name="test_data.csv",
            problem_type="regression"
        )
        
        mock_plot_data = report_data.PlotData(
            name="test_plot",
            description="test description",
            image="<svg>test</svg>"
        )
        
        with mock.patch.object(
            reporting_service, 
            "_create_plot_data", 
            return_value=mock_plot_data
        ):
            with mock.patch.object(
                reporting_service,
                "_create_feature_distribution",
                return_value=None
            ):
                reporting_service.add_dataset("regression", splits)
        
        dataset_id = "regression_test_data.csv"
        assert dataset_id in reporting_service.datasets
        
        dataset = reporting_service.datasets[dataset_id]
        
        for split_id, stats in dataset.split_target_stats.items():
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats

    def test_add_dataset_clears_cache(self, reporting_service):
        """Test that add_dataset clears internal caches."""
        splits = factories.DataSplitsFactory.simple(n_splits=2)
        
        reporting_service._image_cache[("g", "d", "s", "m")] = ("img", {})
        reporting_service._table_cache[("g", "d", "s", "m")] = ({}, {})
        reporting_service._cached_tuned_params = {"param": "value"}
        
        mock_plot_data = report_data.PlotData(
            name="test_plot",
            description="test description",
            image="<svg>test</svg>"
        )
        
        with mock.patch.object(
            reporting_service, 
            "_create_plot_data", 
            return_value=mock_plot_data
        ):
            with mock.patch.object(
                reporting_service,
                "_create_feature_distribution",
                return_value=None
            ):
                reporting_service.add_dataset("test_group", splits)
        
        assert reporting_service._image_cache == {}
        assert reporting_service._table_cache == {}
        assert reporting_service._cached_tuned_params == {}

    def test_add_experiment_clears_cache(self, reporting_service, mock_algorithm):
        """Test that add_experiment clears internal caches."""
        reporting_service.set_context(
            "test_group", ("test_data.csv", None), 0, 
            feature_names=["f1", "f2"],
            algorithm_names=["ridge"]
        )
        
        reporting_service._image_cache[("g", "d", "s", "m")] = ("img", {
            "method": "test", "is_test": False
        })
        reporting_service._table_cache[("g", "d", "s", "m")] = ({}, {
            "method": "test", "is_test": False
        })
        reporting_service._cached_tuned_params = {"param": "value"}
        
        reporting_service.add_experiment(mock_algorithm)
        assert reporting_service._image_cache == {}
        assert reporting_service._table_cache == {}
        assert reporting_service._cached_tuned_params == {}

    def test_add_experiment_no_algorithms(self, reporting_service, mock_algorithm):
        """Test add_experiment with no algorithms in context."""
        reporting_service.set_context(
            "test_group", ("test_data.csv", None), 0,
            feature_names=["f1", "f2"],
            algorithm_names=[]
        )
        
        reporting_service.add_experiment(mock_algorithm)
        
        experiment_id = "_test_group_test_data.csv"
        assert experiment_id in reporting_service.experiments
        assert reporting_service.experiments[experiment_id].algorithm == []

    def test_add_experiment_one_algorithm(self, reporting_service, mock_algorithm):
        """Test add_experiment with one algorithm."""
        reporting_service.set_context(
            "test_group", ("test_data.csv", None), 0,
            feature_names=["f1", "f2"],
            algorithm_names=["ridge"]
        )
        
        reporting_service.add_experiment(mock_algorithm)
        
        experiment_id = "ridge_test_group_test_data.csv"
        experiment = reporting_service.experiments[experiment_id]
        assert len(experiment.algorithm) == 1
        assert experiment.algorithm[0] == "ridge Display"
        assert experiment.dataset == "test_group_test_data.csv"

    def test_add_experiment_two_algorithms(self, reporting_service, mock_algorithm):
        """Test add_experiment with two algorithms."""
        reporting_service.set_context(
            "test_group", ("test_data.csv", None), 0,
            feature_names=["f1", "f2"],
            algorithm_names=["ridge", "lasso"]
        )
        
        reporting_service.add_experiment(mock_algorithm)
        
        experiment_id = "ridge_lasso_test_group_test_data.csv"
        experiment = reporting_service.experiments[experiment_id]
        assert len(experiment.algorithm) == 2
        assert experiment.algorithm[0] == "ridge Display"
        assert experiment.algorithm[1] == "lasso Display"

    def test_add_experiment_groups_no_groups(self, reporting_service):
        """Test add_experiment_groups with empty group list."""
        reporting_service.add_experiment_groups([])
        assert len(reporting_service.experiment_groups) == 0

    def test_add_experiment_groups_one_group(self, reporting_service, mock_data_manager):
        """Test add_experiment_groups with one group."""
        Group = namedtuple("Group", ["name", "description", "datasets"])
        group = Group(
            name="classification",
            description="Classification experiments",
            datasets=[("iris.csv", None)]
        )
        
        reporting_service.add_data_manager("classification", mock_data_manager)
        
        reporting_service.test_scores["classification"]["iris.csv"][0]["columns"] = ["Algorithm", "Accuracy"]
        reporting_service.test_scores["classification"]["iris.csv"][0]["rows"] = [["RF", "0.95"]]
        
        reporting_service.add_experiment_groups([group])
        
        assert len(reporting_service.experiment_groups) == 1
        
        exp_group = reporting_service.experiment_groups[0]
        assert exp_group.name == "classification"
        assert exp_group.description == "Classification experiments"
        assert len(exp_group.datasets) == 1
        assert "iris" in exp_group.datasets

    def test_add_experiment_groups_two_groups(self, reporting_service, mock_data_manager):
        """Test add_experiment_groups with two groups."""
        Group = namedtuple("Group", ["name", "description", "datasets"])
        group1 = Group(
            name="classification",
            description="Classification experiments",
            datasets=[("iris.csv", None)]
        )
        group2 = Group(
            name="regression",
            description="Regression experiments",
            datasets=[("housing.csv", None)]
        )
        
        reporting_service.add_data_manager("classification", mock_data_manager)
        reporting_service.add_data_manager("regression", mock_data_manager)
        
        reporting_service.test_scores["classification"]["iris.csv"][0]["columns"] = ["Algorithm", "Accuracy"]
        reporting_service.test_scores["classification"]["iris.csv"][0]["rows"] = [["RF", "0.95"]]
        reporting_service.test_scores["regression"]["housing.csv"][0]["columns"] = ["Algorithm", "RMSE"]
        reporting_service.test_scores["regression"]["housing.csv"][0]["rows"] = [["Linear", "2.5"]]
        
        reporting_service.add_experiment_groups([group1, group2])
        
        assert len(reporting_service.experiment_groups) == 2
        
        # Verify first group
        exp_group1 = reporting_service.experiment_groups[0]
        assert exp_group1.name == "classification"
        assert "iris" in exp_group1.datasets
        
        # Verify second group
        exp_group2 = reporting_service.experiment_groups[1]
        assert exp_group2.name == "regression"
        assert "housing" in exp_group2.datasets

    def test_set_and_get_context(self, reporting_service):
        reporting_service.set_context(
            "test_group", "test_dataset", 0,
            feature_names=["f1", "f2"],
            algorithm_names=["ridge"]
        )
        
        group, dataset, split, features, algorithms = reporting_service.get_context()
        
        assert group == "test_group"
        assert dataset == "test_dataset"
        assert split == 0
        assert features == ["f1", "f2"]
        assert algorithms == ["ridge"]

    def test_clear_context(self, reporting_service):
        reporting_service.set_context("test_group", "test_dataset", 0)
        reporting_service.clear_context()
        
        with pytest.raises(ValueError, match="No context set"):
            reporting_service.get_context()

    def test_get_context_without_setting(self, reporting_service):
        """Test getting context without setting it first."""
        with pytest.raises(ValueError, match="No context set"):
            reporting_service.get_context()
