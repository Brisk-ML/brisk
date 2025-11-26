"""Unit tests for Experiment"""
import pytest
from sklearn import linear_model
from pathlib import Path

from brisk.configuration.experiment import Experiment

from tests.utils.factories import AlgorithmFactory

class TestExperiment:
    def test_initalization(self, tmp_path):
        ridge_wrapper = AlgorithmFactory.simple()
        experiment = Experiment(
            group_name="test_group",
            workflow="test_workflow",
            algorithms={"model": ridge_wrapper},
            dataset_path=tmp_path / "datasets" / "data.csv",
            workflow_args={},
            split_index=0,
        )
        assert experiment.name == "test_group_ridge"
        assert experiment.dataset_name == ("data", None)
        assert isinstance(
            experiment.algorithm_kwargs["model"], linear_model.Ridge
        )
        assert experiment.algorithm_names == ["ridge"]
        assert list(experiment.workflow_attributes.keys()) == ["model"]
        assert isinstance(
            experiment.workflow_attributes["model"], linear_model.Ridge
        )

    def test_no_algorithm(self, tmp_path):
        """Must provide at least one algorithm."""
        with pytest.raises(ValueError):
            experiment = Experiment(
                group_name="test_group",
                workflow="test_workflow",
                algorithms={},
                dataset_path=tmp_path / "datasets" / "data.csv",
                workflow_args={},
                split_index=0,
                table_name=None,
                categorical_features=["species"]
            )
    
    def test_three_algorithms(self, tmp_path):
        """Test we handle a variable number of algorithms."""
        ridge_wrapper = AlgorithmFactory.simple()
        linear_wrapper = AlgorithmFactory.full(
            "linear", "Linear Regression", linear_model.LinearRegression,
            {}, {}
        )
        lasso_wrapper = AlgorithmFactory.full(
            "lasso", "LASSO Regression", linear_model.Lasso
        )
        experiment = Experiment(
            group_name="test_group",
            workflow="test_workflow",
            algorithms={
                "model": ridge_wrapper,
                "model2": linear_wrapper,
                "model3": lasso_wrapper
            },
            dataset_path=tmp_path / "datasets" / "data.csv",
            workflow_args={},
            split_index=0,
            table_name=None,
            categorical_features=["species"]
        )
        assert experiment.name == "test_group_ridge_linear_lasso"
        assert isinstance(
            experiment.algorithm_kwargs["model"], linear_model.Ridge
        )
        assert isinstance(
            experiment.algorithm_kwargs["model2"], linear_model.LinearRegression
        )
        assert isinstance(
            experiment.algorithm_kwargs["model3"], linear_model.Lasso
        )
        assert experiment.algorithm_names == ["ridge", "linear", "lasso"]
        assert list(experiment.workflow_attributes.keys()) == [
            "model", "model2", "model3"
        ]
        assert isinstance(
            experiment.workflow_attributes["model"], linear_model.Ridge
        )
        assert isinstance(
            experiment.workflow_attributes["model2"],
            linear_model.LinearRegression
        )
        assert isinstance(
            experiment.workflow_attributes["model3"], linear_model.Lasso
        )

    def test_dataset_name_tuple(self, tmp_path):
        ridge_wrapper = AlgorithmFactory.simple()
        experiment = Experiment(
            group_name="test_group",
            workflow="test_workflow",
            algorithms={"model": ridge_wrapper},
            dataset_path=tmp_path / "datasets" / "data.db",
            workflow_args={},
            split_index=0,
            table_name="table_in_database"
        )
        assert experiment.dataset_name == ("data", "table_in_database")
    
    def test_str_dataset_path(self, tmp_path):
        """If dataset_path is a string convert to a Path instance"""
        ridge_wrapper = AlgorithmFactory.simple()
        experiment = Experiment(
            group_name="test_group",
            workflow="test_workflow",
            algorithms={"model": ridge_wrapper},
            dataset_path=str(tmp_path / "datasets" / "data.csv"),
            workflow_args={},
            split_index=0
        )
        assert isinstance(experiment.dataset_path, Path)
