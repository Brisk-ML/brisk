"""Unit tests for Configuration"""
from unittest.mock import patch

import pytest

from brisk.configuration.experiment_group import ExperimentGroup

from tests.utils.factories import ConfigurationFactory

# Prevent fixtures from raising linter warning
# pylint: disable = W0621, W0613, C0301

class TestConfiguration():
    """Unit tests for Configuration."""
    def test_initialization_minimum_args(self):
        """Test configuration initialization; minimum args."""
        configuration = ConfigurationFactory.simple(
            algorithms=["linear", "ridge"]
        )
        assert configuration.default_algorithms == ["linear", "ridge"]
        assert configuration.experiment_groups == []
        assert configuration.categorical_features == {}
        assert configuration.default_workflow_args == {}
        assert configuration.default_workflow == "test_workflow"

    def test_initialization_categorical_features(self):
        configuration = ConfigurationFactory.full(
            categorical_features = {"categorical": ["category"]}
        )
        assert configuration.default_algorithms == ["linear"]
        assert configuration.categorical_features == {"categorical": ["category"]}
        assert configuration.experiment_groups == []
        assert configuration.default_workflow_args == {}

    def test_initialization_workflow_args(self):
        configuration = ConfigurationFactory.full(
            workflow_args = {"kfold": 5}
        )
        assert configuration.default_algorithms == ["linear"]
        assert configuration.categorical_features == {}
        assert configuration.experiment_groups == []
        assert configuration.default_workflow_args == {"kfold": 5}

    def test_initialization_algorithm_groups(self):
        configuration = ConfigurationFactory.full(
            algorithms = [
                ["linear", "ridge"], ["linear", "elasticnet"]
            ]
        )
        assert configuration.default_algorithms == [
            ["linear", "ridge"], ["linear", "elasticnet"]
        ]
        assert configuration.categorical_features == {}
        assert configuration.experiment_groups == []
        assert configuration.default_workflow_args == {}

    @patch("brisk.configuration.experiment_group.ExperimentGroup._validate_datasets")
    def test_add_experiment_group(self, mock_validate_datasets):
        """Test adding experiment group with defaults"""
        mock_validate_datasets.return_value = None
        configuration = ConfigurationFactory.simple()
        configuration.add_experiment_group(
            name="test_group",
            datasets=["regression.csv"]
        )

        group = configuration.experiment_groups[0]
        assert group.name == "test_group"
        assert group.datasets == [("regression.csv", None)]
        assert group.algorithms == ["linear"]
        assert group.data_config == {}
        assert group.algorithm_config is None
        assert group.description == "No description set."
        assert group.workflow_args == {}

    @patch("brisk.configuration.experiment_group.ExperimentGroup._validate_datasets")
    def test_add_experiment_group_custom_algorithms(self, mock_validate_datasets):
        """Test adding experiment group with custom algorithms"""
        mock_validate_datasets.return_value = None
        algorithm_config = {"elasticnet": {"alpha": 0.5}}
        configuration = ConfigurationFactory.simple(
            "custom_group", ["elasticnet"]
        )
        configuration.add_experiment_group(
            name="custom_group",
            datasets=["regression.csv"],
            algorithms=["elasticnet"],
            algorithm_config=algorithm_config,
            description="This is a test description"
        )

        group = configuration.experiment_groups[0]
        assert group.name == "custom_group"
        assert group.datasets == [("regression.csv", None)]
        assert group.algorithms == ["elasticnet"]
        assert group.algorithm_config == algorithm_config
        assert group.description == "This is a test description"
        assert group.workflow_args == {}

    @patch("brisk.configuration.experiment_group.ExperimentGroup._validate_datasets")
    def test_add_experiment_group_passes_defaults(self, mock_validate_datasets):
        """Test Configuration passes defaults to ExperimentGroup."""
        mock_validate_datasets.return_value = None
        configuration = ConfigurationFactory.full(
            workflow="My Workflow",
            algorithms=["linear", "rf"],
            workflow_args={"repeats": 5}
        )
        configuration.add_experiment_group(
            name="group1", description="a test group", datasets=["test.csv"]
        )

        group = configuration.experiment_groups[0]
        assert group.name == "group1"
        assert group.workflow == "My Workflow"
        assert group.datasets == [("test.csv", None)]
        assert group.algorithms == ["linear", "rf"]
        assert group.algorithm_config is None
        assert group.description == "a test group"
        assert group.workflow_args == {"repeats": 5}

    @patch("brisk.configuration.experiment_group.ExperimentGroup._validate_datasets")
    def test_duplicate_name(self, mock_validate_datasets):
        """Test adding experiment group with duplicate name"""
        mock_validate_datasets.return_value = None
        configuration = ConfigurationFactory.simple()
        # Add first group
        configuration.add_experiment_group(
            name="test_group",
            datasets=["regression.csv"]
        )

        # Attempt to add duplicate
        with pytest.raises(ValueError, match="already exists"):
            configuration.add_experiment_group(
                name="test_group",
                datasets=["regression.csv"]
            )

    @patch("brisk.configuration.experiment_group.ExperimentGroup._validate_datasets")
    def test_add_experiment_workflow_args_missing_key(
            self, mock_validate_datasets
    ):
        """Test adding experiment group with workflow args"""
        mock_validate_datasets.return_value = None
        configuration = ConfigurationFactory.simple()
        with pytest.raises(
            ValueError,
            match="workflow_args must have the same keys as defined in default_workflow_args"
        ):
            configuration.add_experiment_group(
                name="test_group",
                datasets=["regression.csv"],
                workflow_args={"kfold": 10, "metrics": ["MAE", "R2"]}
            )

    @patch("brisk.configuration.experiment_group.ExperimentGroup._validate_datasets")
    def test_check_name_exists(self, mock_validate_datasets):
        """Test check_name_exists method"""
        mock_validate_datasets.return_value = None
        configuration = ConfigurationFactory.simple()
        configuration.experiment_groups = [
            ExperimentGroup(
                name="group", workflow="regression_workflow",
                datasets=["regression.csv"]
            ),
            ExperimentGroup(
                name="group_2", workflow="regression_workflow",
                datasets=["regression.csv"]
            ),
            ExperimentGroup(
                name="group_3", workflow="regression_workflow",
                datasets=["regression.csv"]
            )
        ]
        with pytest.raises(ValueError, match="already exists"):
            configuration.add_experiment_group(
                name="group", datasets=["test.csv"]
            )

        with pytest.raises(ValueError, match="already exists"):
            configuration.add_experiment_group(
                name="group_2", datasets=["test.csv"]
            )

        with pytest.raises(ValueError, match="already exists"):
            configuration.add_experiment_group(
                name="group_3", datasets=["test.csv"]
            )

        configuration.add_experiment_group(
            name="group_4", datasets=["test.csv"]
        )

    @patch("brisk.configuration.experiment_group.ExperimentGroup._validate_datasets")
    def test_check_datasets_type_error_dict(self, mock_validate_datasets):
        datasets_dict = [{"path_to_data": "table_name"}]
        configuration = ConfigurationFactory.simple()
        with pytest.raises(
            TypeError,
            match="datasets must be a list containing strings and/or tuples "
        ):
            configuration.add_experiment_group(
                name="group", datasets=datasets_dict
            )

    @patch("brisk.configuration.experiment_group.ExperimentGroup._validate_datasets")
    def test_check_datasets_type_error_lists(self, mock_validate_datasets):
        datasets_list = [["path", "to", "data"], ["more", "data"]]
        configuration = ConfigurationFactory.simple()
        with pytest.raises(
            TypeError,
            match="datasets must be a list containing strings and/or tuples "
        ):
            configuration.add_experiment_group(
                name="group", datasets=datasets_list
            )

    @patch("brisk.configuration.experiment_group.ExperimentGroup._validate_datasets")
    def test_check_datasets_type_error_correct(self, mock_validate_datasets):
        datasets_correct = ["path_to_data", ("file_path", "table_name")]
        configuration = ConfigurationFactory.simple()
        configuration.add_experiment_group(
            name="group", datasets=datasets_correct
        )

    @patch("brisk.configuration.experiment_group.ExperimentGroup._validate_datasets")
    def test_convert_datasets_to_tuple_mix_input(self, mock_validate_datasets):
        configuration = ConfigurationFactory.simple()
        datasets = [
            "data.csv", ("mixed_features.db", "mixed_features_regression")
        ]
        configuration.add_experiment_group(
            name="group", datasets=datasets
        )
        formated_datasets = configuration.experiment_groups[0].datasets
        assert formated_datasets == [
            ("data.csv", None),
            ("mixed_features.db", "mixed_features_regression")
        ]

    @patch("brisk.configuration.experiment_group.ExperimentGroup._validate_datasets")
    def test_convert_datasets_to_tuple_input(self, mock_validate_datasets):
        configuration = ConfigurationFactory.simple()
        datasets = [
            ("data.db", "data_table1"),
            ("data.db", "data_table2"),
        ]
        configuration.add_experiment_group(
            name="group", datasets=datasets
        )
        formated_datasets = configuration.experiment_groups[0].datasets
        assert formated_datasets == [
            ("data.db", "data_table1"),
            ("data.db", "data_table2"),
        ]

    @patch("brisk.configuration.experiment_group.ExperimentGroup._validate_datasets")
    def test_convert_datasets_to_tuple_str_input(self, mock_validate_datasets):
        configuration = ConfigurationFactory.simple()
        datasets = [
            "data.csv", "test.csv"
        ]
        configuration.add_experiment_group(
            name="group", datasets=datasets
        )
        formated_datasets = configuration.experiment_groups[0].datasets
        assert formated_datasets == [
            ("data.csv", None),
            ("test.csv", None)
        ]
