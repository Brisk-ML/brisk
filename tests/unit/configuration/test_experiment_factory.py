"""Unit tests for ExperimentFactory"""
from pathlib import Path

import pytest

from brisk.configuration.experiment_factory import ExperimentFactory

from tests.utils.factories import AlgorithmFactory, ExperimentGroupFactory

# pylint: disable=W0621

@pytest.fixture
def factory():
    """Create ExperimentFactory instance."""
    algorithm_config = AlgorithmFactory.collection()
    return ExperimentFactory(algorithm_config, {})


@pytest.fixture
def factory_categorical():
    """Create ExperimentFactory instance with categorical feature name map."""
    algorithm_config = AlgorithmFactory.collection()
    return ExperimentFactory(algorithm_config, {
        "categorical.csv": ["category"]
    })


class TestExperimentFactory:
    """Unit tests for ExperimentFactory"""
    def test_init_error_none(self):
        """Raise error if algorithm_config is not an AlgorithmCollection"""
        with pytest.raises(
            TypeError,
            match="algorithm_config must be an AlgorithmCollection"
        ):
            ExperimentFactory(None, {})

    def test_init_error_list(self):
        """Raise error if algorithm_config is not an AlgorithmCollection"""
        with pytest.raises(
            TypeError,
            match="algorithm_config must be an AlgorithmCollection"
        ):
            ExperimentFactory([], {})

    def test_init_error_dict(self):
        """Raise error if algorithm_config is not an AlgorithmCollection"""
        with pytest.raises(
            TypeError,
            match="algorithm_config must be an AlgorithmCollection"
        ):
            ExperimentFactory({}, {})

    def test_init_error_list_of_wrappers(self):
        """Raise error if algorithm_config is not an AlgorithmCollection"""
        with pytest.raises(
            TypeError,
            match="algorithm_config must be an AlgorithmCollection"
        ):
            algorithm_list = [
                AlgorithmFactory.linear(),
                AlgorithmFactory.ridge()
            ]
            ExperimentFactory(algorithm_list, {})

    def test_single_algorithm(self, factory):
        """Test creation of experiment with single algorithm."""
        group = ExperimentGroupFactory.simple(
            name="test",
            workflow="regression_workflow",
            datasets=["regression.csv"],
            algorithms=["linear"]
        )

        experiments = factory.create_experiments(group, 1)
        assert len(experiments) == 1
        exp = experiments[0]
        assert exp.group_name == "test"
        assert len(exp.algorithms) == 1
        assert "model" in exp.algorithms
        assert exp.dataset_path == Path("datasets/regression.csv")
        assert exp.workflow_args is None
        assert exp.table_name is None
        assert exp.categorical_features is None

    def test_multiple_separate_algorithms(self, factory):
        """Test creation of separate experiments for multiple algorithms."""
        group = ExperimentGroupFactory.simple(
            name="test",
            workflow="regression_workflow",
            datasets=["regression.csv"],
            algorithms=["linear", "ridge"]
        )

        experiments = factory.create_experiments(group, 1)
        assert len(experiments) == 2

        # Linear Experiment
        linear_exp = experiments[0]
        assert linear_exp.group_name == "test"
        assert len(linear_exp.algorithms) == 1
        assert "model" in linear_exp.algorithms
        assert linear_exp.dataset_path == Path("datasets/regression.csv")
        assert linear_exp.workflow_args is None
        assert linear_exp.table_name is None
        assert linear_exp.categorical_features is None

        # Ridge Experiment
        ridge_exp = experiments[1]
        assert ridge_exp.group_name == "test"
        assert len(ridge_exp.algorithms) == 1
        assert "model" in ridge_exp.algorithms
        assert ridge_exp.dataset_path == Path("datasets/regression.csv")
        assert ridge_exp.workflow_args is None
        assert ridge_exp.table_name is None
        assert ridge_exp.categorical_features is None

    def test_combined_algorithms(self, factory):
        """Test creation of single experiment with multiple algorithms."""
        group = ExperimentGroupFactory.simple(
            name="test",
            workflow="regression_workflow",
            datasets=["regression.csv"],
            algorithms=[["linear", "ridge"]]
        )

        experiments = factory.create_experiments(group, 1)
        assert len(experiments) == 1
        exp = experiments[0]
        assert len(exp.algorithms) == 2
        assert "model" in exp.algorithms
        assert "model2" in exp.algorithms
        assert exp.dataset_path == Path("datasets/regression.csv")
        assert exp.workflow_args is None
        assert exp.table_name is None
        assert exp.categorical_features is None

    def test_multiple_datasets(self, factory):
        """Test creation of experiments for multiple datasets."""
        group = ExperimentGroupFactory.simple(
            name="test",
            workflow="regression_workflow",
            datasets=["regression.csv", "group.csv"],
            algorithms=["linear"]
        )

        experiments = factory.create_experiments(group, 1)
        assert len(experiments) == 2
        # Linear Experiment
        exp = experiments[0]
        assert exp.group_name == "test"
        assert len(exp.algorithms) == 1
        assert "model" in exp.algorithms
        assert exp.dataset_path == Path("datasets/regression.csv")
        assert exp.workflow_args is None
        assert exp.table_name is None
        assert exp.categorical_features is None

    def test_algorithm_config(self, factory):
        """Test application of algorithm configuration."""
        group = ExperimentGroupFactory.simple(
            name="test",
            workflow="regression_workflow",
            datasets=["regression.csv"],
            algorithms=["ridge"],
            algorithm_config={
                "ridge": {
                    "alpha": [0.1, 0.2, 0.3]
                }
            }
        )

        experiments = factory.create_experiments(group, 1)
        exp = experiments[0]

        # Check hyperparameter grid was updated
        assert "alpha" in exp.algorithms["model"].hyperparam_grid
        assert exp.algorithms["model"].hyperparam_grid["alpha"] == [
            0.1, 0.2, 0.3
        ]

        # Check default params weren't modified
        assert exp.algorithms["model"].default_params["alpha"] == 1.0
        assert exp.algorithms["model"].default_params["max_iter"] == 10000

    def test_invalid_algorithm(self, factory):
        """Test handling of invalid algorithm name."""
        group = ExperimentGroupFactory.simple(
            name="test",
            workflow="regression_workflow",
            datasets=["regression.csv"],
            algorithms=["invalid_algo"]
        )

        with pytest.raises(KeyError, match="No algorithm found with name: "):
            factory.create_experiments(group, 1)

    def test_mixed_algorithm_groups(self, factory):
        """Test handling of mixed single and grouped algorithms."""
        group = ExperimentGroupFactory.simple(
            name="test",
            workflow="regression_workflow",
            datasets=["regression.csv"],
            algorithms=["linear", ["ridge", "lasso"]]
        )

        experiments = factory.create_experiments(group, 1)
        assert len(experiments) == 2

        # Check single algorithm experiment
        single = next(exp for exp in experiments if len(exp.algorithms) == 1)
        assert "model" in single.algorithms
        assert single.dataset_path == Path("datasets/regression.csv")
        assert single.workflow_args is None
        assert single.table_name is None
        assert single.categorical_features is None

        # Check grouped algorithm experiment
        grouped = next(exp for exp in experiments if len(exp.algorithms) == 2)
        assert "model" in grouped.algorithms
        assert "model2" in grouped.algorithms
        assert grouped.dataset_path == Path("datasets/regression.csv")
        assert grouped.workflow_args is None
        assert grouped.table_name is None
        assert grouped.categorical_features is None

    def test_create_experiment_categorical_features(self, factory_categorical):
        group = ExperimentGroupFactory.simple(
            name="test",
            workflow="regression_workflow",
            datasets=["categorical.csv"],
            algorithms=["linear"]
        )
        experiments = factory_categorical.create_experiments(group, 1)
        assert len(experiments) == 1
        exp = experiments[0]
        assert exp.group_name == "test"
        assert len(exp.algorithms) == 1
        assert "model" in exp.algorithms
        assert exp.dataset_path == Path("datasets/categorical.csv")
        assert exp.workflow_args is None
        assert exp.table_name is None
        assert exp.categorical_features == ["category"]

    def test_create_experiment_sql(self, factory):
        group = ExperimentGroupFactory.simple(
            name="test",
            workflow="regression_workflow",
            datasets=[("test_data.db", "regression")],
            algorithms=["linear"]
        )
        experiments = factory.create_experiments(group, 1)
        assert len(experiments) == 1
        exp = experiments[0]
        assert exp.group_name == "test"
        assert len(exp.algorithms) == 1
        assert "model" in exp.algorithms
        assert exp.dataset_path == Path("datasets/test_data.db")
        assert exp.table_name == "regression"
        assert exp.categorical_features is None

    def test_normalize_algorithms_list(self, factory):
        """Test normalizing algorithms."""
        normalized = factory.normalize_algorithms(["linear", "ridge"])
        assert normalized == [["linear"], ["ridge"]]

    def test_normalize_algorithms_nested_list(self, factory):
        normalized = factory.normalize_algorithms([["linear", "ridge"]])
        assert normalized == [["linear", "ridge"]]

    def test_normalize_algorithms_str_list(self, factory):
        normalized = factory.normalize_algorithms(
            ["linear", ["ridge", "elasticnet"]]
        )
        assert normalized == [["linear"], ["ridge", "elasticnet"]]

    def test_normalize_algorithms_mixed(self, factory):
        normalized = factory.normalize_algorithms(
            ["linear", "ridge", "elasticnet", ["lasso", "ridge"], ["ridge"]]\
        )
        assert normalized == [
            ["linear"], ["ridge"], ["elasticnet"], ["lasso", "ridge"], ["ridge"]
        ]

    def test_normalize_algorithms_error_str(self, factory):
        """Test error is thrown if algorithms is not a list."""
        with pytest.raises(TypeError, match="algorithms must be a list, got"):
            factory.normalize_algorithms("linear, ridge, elasticnet")

    def test_normalize_algorithms_error_set(self, factory):
        with pytest.raises(TypeError, match="algorithms must be a list, got"):
            factory.normalize_algorithms({"linear", "ridge", "elasticnet"})

    def test_normalize_algorithms_error_int(self, factory):
        with pytest.raises(
            TypeError,
            match="algorithms must contain strings or lists of strings, got"
        ):
            factory.normalize_algorithms(["linear", 1, "elasticnet"])

    def test_normalize_algorithms_error_nested_int(self, factory):
        with pytest.raises(
            TypeError,
            match="nested algorithm lists must contain strings, got"
        ):
            factory.normalize_algorithms(["linear", ["ridge", 1], "elasticnet"])

    def test_normalize_algorithms_error_nested_set(self, factory):
        with pytest.raises(
            TypeError,
            match="algorithms must contain strings or lists of strings, got"
        ):
            factory.normalize_algorithms(
                ["linear", ["ridge", "elasticnet"], {"linear", "ridge"}]
            )
