"""Unit tests for ExperimentGroup."""
from pathlib import Path
import textwrap

import pytest

from brisk.configuration.experiment_group import ExperimentGroup

from tests.utils.factories import ExperimentGroupFactory

class TestExperimentGroup:
    def test_valid_creation(self):
        """Test creation with valid parameters"""  
        valid_group = ExperimentGroupFactory.simple()
        assert valid_group.name == "test_group"
        assert valid_group.datasets == ["test.csv"]
        assert valid_group.workflow == "test_workflow"
        assert valid_group.algorithms == ["linear"]
        assert valid_group.algorithm_config == {}
        assert valid_group.description == ""
        assert valid_group.workflow_args is None
        assert valid_group.dataset_paths == [
            (Path("./datasets/test.csv"), None)
        ]

    def test_name_empty_str_error(self):
        """Test creation with invalid name"""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            ExperimentGroup(name="", workflow="regression_workflow", datasets=["regression.csv"])
        
    def test_name_none_error(self):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            ExperimentGroup(name=None, workflow="regression_workflow", datasets=["regression.csv"])

    def test_name_int_error(self):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            ExperimentGroup(name=1, workflow="regression_workflow", datasets=["regression.csv"])

    def test_missing_datasets(self):
        """Test creation with missing dataset"""
        with pytest.raises(ValueError, match="At least one dataset must be specified"):
            ExperimentGroup(
                name="test_group",
                workflow="regression_workflow",
                datasets=[],
                algorithms=["linear", "ridge"]
            )

    def test_invalid_data_config(self):
        """Test creation with invalid data configuration"""
        with pytest.raises(ValueError, match="Invalid DataManager parameters"):
            ExperimentGroupFactory.with_preprocessing(
                data_config={"invalid_param": 1.0}
            )

    @pytest.mark.parametrize("data_config", [
        {"test_size": 0.2},
        {"split_method": "kfold", "n_splits": 5},
        {}
    ])
    def test_valid_data_configs(self, data_config):
        """Test various valid data configurations"""
        group = ExperimentGroupFactory.with_preprocessing(
            data_config=data_config
        )
        assert group.data_config == data_config

    def test_long_description_wrap(self):
        """Test long description is wrapped"""
        group = ExperimentGroupFactory.simple(
            description="This is a long description that should be wrapped over multiple lines since nobody wants to scroll across the screen to read this useless message."
        )
        expected_string = textwrap.dedent("""
        This is a long description that should be wrapped over
        multiple lines since nobody wants to scroll across the
        screen to read this useless message.
        """).strip()
        assert group.description == expected_string

    def test_int_description_error(self):
        """Test invalid description raises ValueError"""
        with pytest.raises(ValueError, match="Description must be a string"):
            ExperimentGroupFactory.simple(
                name="test",
                workflow="regression_workflow",
                datasets=["regression.csv"],
                description=1
            )

    def test_list_description_error(self):
        with pytest.raises(ValueError, match="Description must be a string"):
            ExperimentGroupFactory.simple(
                name="test",
                workflow="regression_workflow",
                datasets=["regression.csv"],
                description=["A description", "that is not a string"]
            )

    def test_invalid_workflow_args(self):
        with pytest.raises(ValueError, match="workflow_args must be a dict"):
            ExperimentGroupFactory.simple(
                name="test_invalid_args",
                workflow="regression_workflow",
                datasets=["regression.csv"],
                workflow_args=["arg1"]
            )
