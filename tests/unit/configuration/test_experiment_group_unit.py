"""Unit tests for ExperimentGroup."""

import pytest

from brisk.configuration.project import ProjectRootContext
from brisk.configuration.experiment_group import ExperimentGroup

from tests.utils.factories import ExperimentGroupFactory

class TestExperimentGroupUnit:
    def test_initalization(self, tmp_path):
        datasets=["data.csv"]
        ExperimentGroupFactory.create_dataset_files(tmp_path, datasets)
        with ProjectRootContext(tmp_path):
            group = ExperimentGroup(
                name="test_group",
                datasets=datasets,
                workflow="test_workflow",
                algorithms=["ridge", "linear"],
                data_config={"split_method": "kfold"},
                algorithm_config={"ridge": {"alpha": [0.01, 0.1, 0.2]}},
                description="This group is used for unit tests.",
                workflow_args={"cv": 3}
            )
            assert group.dataset_paths == [(
                tmp_path / "datasets" / "data.csv", None
            )]

    def test_no_optionals(self, tmp_path):
        datasets = ["data.csv"]
        ExperimentGroupFactory.create_dataset_files(tmp_path, datasets)
        with ProjectRootContext(tmp_path):
            group = ExperimentGroup(
                name="test_group",
                datasets=datasets,
                workflow="test_workflow",
                algorithms=["ridge", "linear"]
            )
            assert group.dataset_paths == [(
                tmp_path / "datasets" / "data.csv", None
            )]
    
    def test_empty_str_name(self, tmp_path):
        datasets=["data.csv"]
        ExperimentGroupFactory.create_dataset_files(tmp_path, datasets)
        with pytest.raises(ValueError):
            group = ExperimentGroup(
                name="",
                datasets=datasets,
                workflow="test_workflow",
                algorithms=["ridge", "linear"]
            )
    
    def test_nested_algorithm_none_missing(self, tmp_path):
        """Check algorithms in nested lists are found when verifying config"""
        datasets = ["data.csv"]
        algorithm_config = {"ridge": {"alpha": [0.1, 0.2]}}
        ExperimentGroupFactory.create_dataset_files(tmp_path, datasets)
        with ProjectRootContext(tmp_path):
            group = ExperimentGroup(
                name="test_group",
                datasets=datasets,
                workflow="test_workflow",
                algorithms=[["ridge"], ["linear"]],
                algorithm_config=algorithm_config
            )
        assert group.algorithm_config == algorithm_config

    def test_nested_algorithm_missing(self, tmp_path):
        datasets = ["data.csv"]
        algorithm_config = {"ridge": {"alpha": [0.1, 0.2]}}
        ExperimentGroupFactory.create_dataset_files(tmp_path, datasets)
        with ProjectRootContext(tmp_path):
            with pytest.raises(ValueError):
                group = ExperimentGroup(
                    name="test_group",
                    datasets=datasets,
                    workflow="test_workflow",
                    algorithms=[["rf", "linear"]],
                    algorithm_config=algorithm_config
                )

    def test_data_manager_invalid_params(self, tmp_path):
        datasets = ["data.csv"]
        ExperimentGroupFactory.create_dataset_files(tmp_path, datasets)
        with ProjectRootContext(tmp_path):
            with pytest.raises(ValueError):
                group = ExperimentGroup(
                    name="test_group",
                    datasets=datasets,
                    workflow="test_workflow",
                    algorithms=["ridge", "linear"],
                    data_config={"invalid_parameter": 42}
                )

    def test_description_wrapped(self, tmp_path):
        datasets = ["data.csv"]
        description = "This is a very long description that should be wrapped over multiple lines for better readability!"
        ExperimentGroupFactory.create_dataset_files(tmp_path, datasets)
        with ProjectRootContext(tmp_path):
            group = ExperimentGroup(
                name="test_group",
                datasets=datasets,
                workflow="test_workflow",
                algorithms=["ridge", "linear"],
                description=description
            )
        assert all(len(line) < 60 for line in group.description.split("\n"))
