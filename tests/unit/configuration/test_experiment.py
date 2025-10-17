"""Unit tests for Experiment"""
from pathlib import Path

import pytest

from brisk.configuration.experiment import Experiment

from tests.utils.factories import AlgorithmFactory

class TestExperiment:
    """Unit tests for Experiment"""
    def test_valid_single_model(self):
        """Test creation with single model."""
        linear_wrapper = AlgorithmFactory.linear()
        linear_wrapper.default_params = {"fit_intercept": True}
        linear_wrapper.hyperparam_grid = {"fit_intercept": [True, False]}

        single_model = Experiment(
            group_name="test_group",
            workflow="regression_workflow",
            dataset_path=Path("datasets/regression.csv"),
            algorithms={"model": linear_wrapper},
            workflow_args={"metrics": ["MAE", "MSE"]},
            table_name=None,
            categorical_features=None,
            split_index=0
        )

        assert single_model.group_name == "test_group"
        assert len(single_model.algorithms) == 1
        assert single_model.dataset_path == Path("datasets/regression.csv")
        assert single_model.workflow_args == {"metrics": ["MAE", "MSE"]}
        assert single_model.table_name is None
        assert single_model.categorical_features is None
        assert single_model.name == "test_group_linear"
        assert single_model.dataset_name == ("regression", None)
        assert single_model.algorithm_names == ["linear"]
        workflow_attrs = single_model.workflow_attributes
        assert set(workflow_attrs.keys()) == {"metrics", "model"}
        assert workflow_attrs["metrics"] == ["MAE", "MSE"]

    def test_valid_multiple_models(self):
        """Test creation with multiple models."""
        linear_wrapper = AlgorithmFactory.linear()
        rf_wrapper = AlgorithmFactory.random_forest()
        multiple_models = Experiment(
            group_name="test_group",
            workflow="regression_workflow",
            dataset_path=Path("datasets/regression.csv"),
            algorithms={
                "model": linear_wrapper,
                "model2": rf_wrapper
            },
            workflow_args={},
            table_name=None,
            categorical_features=None,
            split_index=0
        )

        assert multiple_models.group_name == "test_group"
        assert len(multiple_models.algorithms) == 2
        assert multiple_models.dataset_path == Path("datasets/regression.csv")
        assert not multiple_models.workflow_args
        assert multiple_models.table_name is None
        assert multiple_models.categorical_features is None
        assert multiple_models.name == "test_group_linear_rf"
        assert multiple_models.dataset_name == ("regression", None)
        assert multiple_models.algorithm_names == ["linear", "rf"]
        workflow_attrs = multiple_models.workflow_attributes
        assert set(workflow_attrs.keys()) == {"model", "model2"}

    def test_sql_table(self):
        """Test creation with sql database."""
        ridge_wrapper = AlgorithmFactory.ridge()
        sql_table = Experiment(
            group_name="sql_table",
            workflow="regression_workflow",
            dataset_path=Path("datasets/test_data.db"),
            algorithms={"model": ridge_wrapper},
            workflow_args={},
            table_name="categorical",
            categorical_features=["category"],
            split_index=0
        )
        assert sql_table.group_name == "sql_table"
        assert len(sql_table.algorithms) == 1
        assert sql_table.dataset_path == Path("datasets/test_data.db")
        assert not sql_table.workflow_args
        assert sql_table.table_name == "categorical"
        assert sql_table.categorical_features == ["category"]
        assert sql_table.name == "sql_table_ridge"
        assert sql_table.dataset_name == ("test_data", "categorical")
        assert sql_table.algorithm_names == ["ridge"]
        workflow_attrs = sql_table.workflow_attributes
        assert set(workflow_attrs.keys()) == {"model"}

    def test_invalid_model_keys(self):
        """Test validation of model naming convention."""
        linear_wrapper = AlgorithmFactory.linear()
        ridge_wrapper = AlgorithmFactory.ridge()

        with pytest.raises(ValueError, match="Single model must use key"):
            Experiment(
                group_name="test",
                workflow="regression_workflow",
                dataset_path="test.csv",
                algorithms={"wrong_key": linear_wrapper},
                workflow_args={},
                table_name=None,
                categorical_features=None,
                split_index=0
            )

        with pytest.raises(ValueError, match="Multiple models must use keys"):
            Experiment(
                group_name="test",
                workflow="regression_workflow",
                dataset_path="test.csv",
                algorithms={
                    "model": linear_wrapper,
                    "wrong_key": ridge_wrapper
                },
                workflow_args={},
                table_name=None,
                categorical_features=None,
                split_index=0
            )

    def test_invalid_group_name(self):
        """Test validation of group name."""
        linear_wrapper = AlgorithmFactory.linear()
        with pytest.raises(ValueError, match="Group name must be a string"):
            Experiment(
                group_name=123,
                workflow="regression_workflow",
                dataset_path="test.csv",
                algorithms={"model": linear_wrapper},
                workflow_args={},
                table_name=None,
                categorical_features=None,
                split_index=0
            )

    def test_invalid_algorithms(self):
        """Test validation of algorithms."""
        linear_wrapper = AlgorithmFactory.linear()
        with pytest.raises(ValueError, match="Algorithms must be a dictionary"):
            Experiment(
                group_name="test",
                workflow="regression_workflow",
                dataset_path="test.csv",
                algorithms=[linear_wrapper],
                workflow_args={},
                table_name=None,
                categorical_features=None,
                split_index=0
            )

    def test_missing_algorithms(self):
        """Test validation of algorithms."""
        with pytest.raises(
            ValueError, match="At least one algorithm must be provided"
        ):
            Experiment(
                group_name="test",
                workflow="regression_workflow",
                dataset_path="test.csv",
                algorithms={},
                workflow_args={},
                table_name=None,
                categorical_features=None,
                split_index=0
            )
