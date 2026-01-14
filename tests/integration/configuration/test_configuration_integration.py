"""Integration tests for Configuration."""
import json
import unittest.mock
import pathlib

import pytest

from brisk.configuration import project
from brisk.configuration import configuration
from brisk.services import bundle, missing 
from brisk.theme import plot_settings

@pytest.fixture()
def mock_rerun_service():
    """Create a mock rerun service that captures configurations."""
    mock = unittest.mock.Mock()
    mock.configs = {
        "package_version": "test",
        "env": {},
        "base_data_manager": None,
        "configuration": {},
        "experiment_groups": [],
        "metrics": [],
        "algorithms": [],
        "evaluators": None,
        "workflows": {},
        "datasets": {},
    }
    
    def add_configuration(config_dict):
        mock.configs["configuration"] = config_dict
    
    def add_experiment_groups(groups_list):
        mock.configs["experiment_groups"] = groups_list
    
    def collect_dataset_metadata(groups_list):
        mock.configs["datasets"] = {"collected": True}
    
    mock.add_configuration.side_effect = add_configuration
    mock.add_experiment_groups.side_effect = add_experiment_groups
    mock.collect_dataset_metadata.side_effect = collect_dataset_metadata
    
    return mock


@pytest.fixture()
def mock_services(mock_rerun_service):
    """Create a service bundle with mocked rerun service."""
    return bundle.ServiceBundle(
        io=missing.MissingServices(),
        logger=missing.MissingServices(),
        metadata=missing.MissingServices(),
        utility=missing.MissingServices(),
        reporting=missing.MissingServices(),
        rerun=mock_rerun_service
    )


@pytest.fixture()
def minimal_config(mock_services):
    """Create minimal configuration with only required parameters."""
    config = configuration.Configuration(
        default_workflow="workflow",
        default_algorithms=["ridge"]
    )
    config.set_services(mock_services)
    return config


@pytest.fixture()
def config_with_optionals(mock_services):
    """Create configuration with all optional parameters."""
    config = configuration.Configuration(
        default_workflow="custom_workflow",
        default_algorithms=["ridge", "lasso", "rf"],
        categorical_features={
            "dataset1.csv": ["cat1", "cat2"],
            ("dataset2.xlsx", "Sheet1"): ["cat3"],
        },
        default_workflow_args={"param1": "value1", "param2": 42},
        plot_settings=plot_settings.PlotSettings(
            file_format="svg",
            width=10,
            height=8
        )
    )
    config.set_services(mock_services)
    return config


@pytest.fixture()
def config_with_experiment_groups(mock_services, tmp_path):
    """Create configuration with experiment groups."""
    config = configuration.Configuration(
        default_workflow="workflow",
        default_algorithms=["ridge", "lasso"],
        categorical_features={
            "data1.csv": ["feature1", "feature2"]
        },
        default_workflow_args={"cv_folds": 5}
    )
    config.set_services(mock_services)
    
    root = pathlib.Path(tmp_path)
    with project.ProjectRootContext(root):
        dataset_dir = root / "datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        filenames = ["data1.csv", "data2.csv", "data3.xlsx", "data4.csv"]
        for name in filenames:
            (dataset_dir / name).touch()

        config.add_experiment_group(
            name="group1",
            datasets=["data1.csv", "data2.csv"],
            description="First test group"
        )
        
        config.add_experiment_group(
            name="group2",
            datasets=[("data3.xlsx", "Sheet1"), "data4.csv"],
            algorithms=["rf", "svm"],
            data_config={"test_size": 0.3, "random_state": 42},
            algorithm_config={"rf": {"n_estimators": 100}},
            description="Advanced group",
            workflow="custom_workflow",
            workflow_args={"cv_folds": 10}
        )
    
    return config


class TestConfigurationIntegration():
    """Test exported params and JSON serialization of these params."""
    def test_minimal_config_is_json_serializable(
        self,
        minimal_config,
        mock_services
    ):
        minimal_config.export_params()

        config_json = mock_services.rerun.configs["configuration"]
        groups_json = mock_services.rerun.configs["experiment_groups"]

        config_str = json.dumps(config_json, indent=2)
        groups_str = json.dumps(groups_json, indent=2)

        deserialized_config = json.loads(config_str)
        deserialized_groups = json.loads(groups_str)
        
        assert isinstance(deserialized_config, dict)
        assert isinstance(deserialized_groups, list)
        assert deserialized_config == config_json

    def test_config_with_optionals_is_json_serializable(
        self,
        config_with_optionals,
        mock_services
    ):
        config_with_optionals.export_params()

        config_json = mock_services.rerun.configs["configuration"]
        groups_json = mock_services.rerun.configs["experiment_groups"]

        config_str = json.dumps(config_json, indent=2)
        groups_str = json.dumps(groups_json, indent=2)

        deserialized_config = json.loads(config_str)
        deserialized_groups = json.loads(groups_str)
        
        assert isinstance(deserialized_config, dict)
        assert isinstance(deserialized_groups, list)
        assert deserialized_config == config_json

    def test_config_with_groups_is_json_serializable(
        self,
        config_with_experiment_groups,
        mock_services
    ):
        config_with_experiment_groups.export_params()

        config_json = mock_services.rerun.configs["configuration"]
        groups_json = mock_services.rerun.configs["experiment_groups"]

        config_str = json.dumps(config_json, indent=2)
        groups_str = json.dumps(groups_json, indent=2)

        deserialized_config = json.loads(config_str)
        deserialized_groups = json.loads(groups_str)
        
        assert isinstance(deserialized_config, dict)
        assert isinstance(deserialized_groups, list)
        assert deserialized_config == config_json

    def test_minimal_config_values(self, minimal_config, mock_services):
        minimal_config.export_params()
        
        config_json = mock_services.rerun.configs["configuration"]

        assert config_json["default_workflow"] == "workflow"
        assert config_json["default_algorithms"] == ["ridge"]
        assert config_json["default_workflow_args"] == {}
        assert config_json["categorical_features"] == []

    def test_config_with_optionals_configuration_values(
        self, config_with_optionals, mock_services
    ):
        """Test that configuration with optionals has correct values."""
        config_with_optionals.export_params()
        
        config_json = mock_services.rerun.configs["configuration"]

        assert config_json["default_workflow"] == "custom_workflow"
        assert config_json["default_algorithms"] == ["ridge", "lasso", "rf"]
        assert config_json["default_workflow_args"] == {
            "param1": "value1",
            "param2": 42
        }
        
    def test_config_with_optionals_categortical_feature_values(
        self, config_with_optionals, mock_services
    ):
        config_with_optionals.export_params()
        
        config_json = mock_services.rerun.configs["configuration"]

        cat_features = config_json["categorical_features"]
        assert isinstance(cat_features, list)
        assert len(cat_features) == 2
        
        # Find the categorical feature items
        dataset1_item = next(
            item for item in cat_features 
            if item["dataset"] == "dataset1.csv"
        )
        dataset2_item = next(
            item for item in cat_features 
            if item["dataset"] == "dataset2.xlsx"
        )
        
        assert dataset1_item["table_name"] is None
        assert dataset1_item["features"] == ["cat1", "cat2"]
        
        assert dataset2_item["table_name"] == "Sheet1"
        assert dataset2_item["features"] == ["cat3"]

    def test_experiment_group_simple_values(
        self, config_with_experiment_groups, mock_services
    ):
        """Test simple experiment group has correct values."""
        config_with_experiment_groups.export_params()
        
        groups_json = mock_services.rerun.configs["experiment_groups"]
        group1 = groups_json[0]
        
        assert group1["name"] == "group1"
        assert group1["description"] == "First test group"
        assert group1["workflow"] == "workflow"
        assert group1["algorithms"] == ["ridge", "lasso"]
        assert group1["data_config"] == {}
        assert group1["algorithm_config"] == {}
        assert group1["workflow_args"] == {"cv_folds": 5}
        
        # Check datasets structure
        assert len(group1["datasets"]) == 2
        assert group1["datasets"][0] == {
            "dataset": "data1.csv",
            "table_name": None
        }
        assert group1["datasets"][1] == {
            "dataset": "data2.csv",
            "table_name": None
        }

    def test_experiment_group_advanced_values(
        self, config_with_experiment_groups, mock_services
    ):
        """Test advanced experiment group has correct values."""
        config_with_experiment_groups.export_params()
        
        groups_json = mock_services.rerun.configs["experiment_groups"]
        group2 = groups_json[1]
        
        assert group2["name"] == "group2"
        assert group2["description"] == "Advanced group"
        assert group2["workflow"] == "custom_workflow"
        assert group2["algorithms"] == ["rf", "svm"]
        assert group2["data_config"] == {
            "test_size": 0.3,
            "random_state": 42
        }
        assert group2["algorithm_config"] == {"rf": {"n_estimators": 100}}
        assert group2["workflow_args"] == {"cv_folds": 10}
        
        # Check datasets with mixed types
        assert len(group2["datasets"]) == 2
        assert group2["datasets"][0] == {
            "dataset": "data3.xlsx",
            "table_name": "Sheet1"
        }
        assert group2["datasets"][1] == {
            "dataset": "data4.csv",
            "table_name": None
        }
   
    def test_empty_categorical_features(self, minimal_config, mock_services):
        """Test that empty categorical features exports correctly."""
        minimal_config.export_params()
        
        config_json = mock_services.rerun.configs["configuration"]
        
        assert config_json["categorical_features"] == []
    
    def test_none_table_name_in_categorical_features(self, mock_services):
        """Test categorical features with None table names."""
        config = configuration.Configuration(
            default_workflow="workflow",
            default_algorithms=["ridge"],
            categorical_features={
                "data.csv": ["cat1", "cat2"]
            }
        )
        config.set_services(mock_services)
        config.export_params()
        
        config_json = mock_services.rerun.configs["configuration"]
        cat_features = config_json["categorical_features"]
        
        assert len(cat_features) == 1
        assert cat_features[0]["dataset"] == "data.csv"
        assert cat_features[0]["table_name"] is None
        assert cat_features[0]["features"] == ["cat1", "cat2"]
    
    def test_dataset_tuple_conversion(self, mock_services, tmp_path):
        """Test that dataset tuples are properly converted to dict format."""
        config = configuration.Configuration(
            default_workflow="workflow",
            default_algorithms=["ridge"]
        )
        config.set_services(mock_services)
        
        root = pathlib.Path(tmp_path)
        with project.ProjectRootContext(root):
            dataset_dir = root / "datasets"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            filenames = ["file1.xlsx", "file2.csv", "file3.db"]
            for name in filenames:
                (dataset_dir / name).touch()

            config.add_experiment_group(
                name="test",
                datasets=[
                    ("file1.xlsx", "Sheet1"),
                    "file2.csv",
                    ("file3.db", "table1")
                ]
            )
        
        config.export_params()
        
        groups_json = mock_services.rerun.configs["experiment_groups"]
        datasets = groups_json[0]["datasets"]
        
        assert datasets[0] == {"dataset": "file1.xlsx", "table_name": "Sheet1"}
        assert datasets[1] == {"dataset": "file2.csv", "table_name": None}
        assert datasets[2] == {"dataset": "file3.db", "table_name": "table1"}
