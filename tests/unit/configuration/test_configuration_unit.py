"""Unit tests for Configuration."""
from unittest import mock

import pytest

from brisk import Configuration
from brisk.theme.plot_settings import PlotSettings
from brisk.configuration.project import ProjectRootContext

# pylint: disable=W0612

@pytest.mark.unit
class TestConfigurationUnit():
    """Unit tests for the Configuration class."""

    def test_initalize_no_optionals(self):
        """Test Configuration can be created with no optionals"""
        config = Configuration(
            "test_workflow",
            ["ridge", "rf"]
        )
        assert config.categorical_features == {}
        assert config.default_workflow_args == {}
        assert isinstance(config.plot_settings, PlotSettings)

    def test_initalize_all_optionals(self):
        """Test Configuration handles all optional arguments"""
        categorical_features = {"data.csv": ["categorical1"]}
        default_workflow_args = {"kf": 5}
        plot_settings = PlotSettings(file_format="svg")
        config = Configuration(
            "test_workflow",
            ["ridge", "rf"],
            categorical_features,
            default_workflow_args,
            plot_settings=plot_settings
        )
        assert config.categorical_features == categorical_features
        assert config.default_workflow_args == default_workflow_args
        assert config.plot_settings == plot_settings

    def test_add_experiment_group_all_optionals(self, tmp_path):
        """Test optional arguments all get passed to the ExperimentGroup"""
        data_config = {"split_method": "shuffle"}
        algorithms = ["linear"]
        algorithm_config = {"linear": {"fit_intercept": True}}
        description = "A test experiment group."
        workflow = "different_workflow"
        workflow_args = {}
        config = Configuration(
            "test_workflow",
            ["ridge", "rf"],
        )

        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "datasets/data.csv").write_text("")
        with ProjectRootContext(tmp_path):
            config.add_experiment_group(
                name="group1",
                datasets=["data.csv"],
                data_config=data_config,
                algorithms=algorithms,
                algorithm_config=algorithm_config,
                description=description,
                workflow=workflow,
                workflow_args=workflow_args
            )

        assert len(config.experiment_groups) == 1
        group = config.experiment_groups[0]
        assert group.data_config == data_config
        assert group.algorithms == algorithms
        assert group.algorithm_config == algorithm_config
        assert group.description == description
        assert group.workflow == workflow
        assert group.workflow_args == workflow_args

    def test_add_experiment_group_no_optionals(self, tmp_path):
        """Test defaults are applied when no optionals are passed"""
        default_algorithms = ["ridge", "rf"]
        default_workflow = "test_workflow"
        config = Configuration(
            default_workflow,
            default_algorithms
        )

        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "datasets/data.csv").write_text("")
        with ProjectRootContext(tmp_path):
            config.add_experiment_group(name="group1", datasets=["data.csv"])

        assert len(config.experiment_groups) == 1
        group = config.experiment_groups[0]
        assert group.name == "group1"
        assert group.algorithms == default_algorithms
        assert group.workflow == default_workflow
        assert group.workflow_args == {}

    def test_add_experiment_group_name_exists(self, tmp_path):
        """Test error is raised when a group name is already taken"""
        config = Configuration(
            "test_workflow",
            ["ridge", "rf"]
        )

        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "datasets/data.csv").write_text("")
        with ProjectRootContext(tmp_path):
            config.add_experiment_group(
                name="group1",
                datasets = ["data.csv"]
            )
            with pytest.raises(ValueError):
                config.add_experiment_group(
                    name="group1",
                    datasets = ["data.csv"]
                )

    def test_add_experiment_group_invalid_dataset_type(self, tmp_path):
        """Test datasets must be str or tuple of strings"""
        config = Configuration(
            "test_workflow",
            ["ridge", "rf"]
        )

        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "datasets/data.csv").write_text("")
        with ProjectRootContext(tmp_path):
            with pytest.raises(TypeError):
                config.add_experiment_group(
                    name="group1",
                    datasets = [["data.csv"]]
                )

    def test_add_experiment_group_workflow_args_invalid_key(self, tmp_path):
        """Test workflow_args keys must be defined in default_workflow_args"""
        config = Configuration(
            "test_workflow",
            ["ridge", "rf"]
        )

        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "datasets/data.csv").write_text("")
        with ProjectRootContext(tmp_path):
            with pytest.raises(ValueError):
                config.add_experiment_group(
                    name="group1",
                    datasets = ["data.csv"],
                    workflow_args = {"this_key_is_missing": True}
                )

    @mock.patch(
        "brisk.configuration.configuration_manager.ConfigurationManager"
    )
    def test_build_correct_params(self, mock_config_manager, tmp_path):
        """Test we create a ConfigurationManager instance"""
        config = Configuration(
            "test_workflow", ["ridge", "rf"]
        )
        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "datasets/data.csv").write_text("")
        with ProjectRootContext(tmp_path):
            config.add_experiment_group(name="group1", datasets=["data.csv"])
        config.set_services(mock.Mock())

        mock_instance = mock.Mock()
        mock_config_manager.return_value = mock_instance

        result = config.build()
        mock_config_manager.assert_called_once_with(
            config.experiment_groups,
            config.categorical_features
        )
        assert result is mock_instance

    @mock.patch(
        "brisk.configuration.configuration_manager.ConfigurationManager"
    )
    def test_build_method_call_order(self, mock_config_manager, tmp_path):
        """Test that build() calls all ConfigurationManager methods in the
        correct sequence.
        """
        config = Configuration(
            "test_workflow", ["ridge", "rf"]
        )
        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "datasets/data.csv").write_text("")
        with ProjectRootContext(tmp_path):
            config.add_experiment_group(name="group1", datasets=["data.csv"])

        config.set_services(mock.Mock())
        config.export_params = mock.Mock()

        mock_instance = mock.Mock()
        mock_config_manager.return_value = mock_instance

        result = config.build()

        expected_calls = [
            mock.call.set_services(config.plot_settings),
            mock.call.load_algorithm_config(),
            mock.call.load_base_data_manager(),
            mock.call.create_data_managers(),
            mock.call.create_experiment_queue(),
            mock.call.create_data_splits(),
            mock.call.create_logfile(),
            mock.call.get_output_structure(),
            mock.call.create_description_map(),
        ]

        mock_instance.assert_has_calls(expected_calls, any_order=False)

    def test_convert_dataset_to_tuple_str(self, tmp_path):
        """Test that string datasets are converted to (dataset, None) tuples."""
        config = Configuration("test_workflow", ["ridge", "rf"])
        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "datasets/data1.csv").write_text("")
        (tmp_path / "datasets/data2.csv").write_text("")

        with ProjectRootContext(tmp_path):
            config.add_experiment_group(
                name="group1",
                datasets=["data1.csv", "data2.csv"]
            )

        group = config.experiment_groups[0]
        assert group.datasets == [("data1.csv", None), ("data2.csv", None)]

    def test_convert_dataset_to_tuple_tuple(self, tmp_path):
        """Test that tuple datasets remain as tuples unchanged."""
        config = Configuration("test_workflow", ["ridge", "rf"])
        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "datasets/data.xlsx").write_text("")

        with ProjectRootContext(tmp_path):
            config.add_experiment_group(
                name="group1",
                datasets=[("data.xlsx", "Sheet1"), ("data.xlsx", "Sheet2")]
            )

        group = config.experiment_groups[0]
        expected = [("data.xlsx", "Sheet1"), ("data.xlsx", "Sheet2")]
        assert group.datasets == expected

    def test_convert_dataset_to_tuple_str_and_tuple(self, tmp_path):
        """Test that mixed string and tuple datasets are handled correctly."""
        config = Configuration("test_workflow", ["ridge", "rf"])
        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "datasets/data1.csv").write_text("")
        (tmp_path / "datasets/data2.xlsx").write_text("")
        (tmp_path / "datasets/data3.csv").write_text("")

        with ProjectRootContext(tmp_path):
            config.add_experiment_group(
                name="group1",
                datasets=[
                    "data1.csv",
                    ("data2.xlsx", "Sheet1"),
                    "data3.csv",
                    ("data2.xlsx", "Sheet2")
                ]
            )

        group = config.experiment_groups[0]
        expected = [
            ("data1.csv", None),
            ("data2.xlsx", "Sheet1"),
            ("data3.csv", None),
            ("data2.xlsx", "Sheet2")
        ]
        assert group.datasets == expected
