"""Unit tests for ConfigurationManager"""
import pytest
from unittest import mock

from brisk.configuration.project import ProjectRootContext
from brisk.theme.plot_settings import PlotSettings
from brisk import ConfigurationManager, DataManager
from brisk.data.preprocessing import (
    ScalingPreprocessor, MissingDataPreprocessor
)

from tests.utils.factories import ExperimentGroupFactory

# pylint: disable=W0621, W0612

@pytest.fixture
def simple_manager(tmp_path, mock_services):
    with ProjectRootContext(tmp_path):
        experiment_groups = [ExperimentGroupFactory.simple(tmp_path)]
        manager = ConfigurationManager(
            experiment_groups,
            {},
        )
    manager.set_services(PlotSettings(), mock_services)
    return manager


@pytest.mark.unit
class TestConfigurationManagerUnit:
    """Unit tests for the ConfigurationManager class."""

    def test_initialize(self, tmp_path):
        """Test ConfigurationManager can be initalized with required args."""
        with ProjectRootContext(tmp_path):
            experiment_groups = [ExperimentGroupFactory.simple(tmp_path)]
            manager = ConfigurationManager(
                experiment_groups,
                {},
            )
        assert manager.experiment_groups == experiment_groups

    def test_set_services(self, simple_manager):
        """Test set_services makes required calls to the service layer."""
        simple_manager.services.io.set_io_settings.assert_called_once()
        simple_manager.services.utility.set_plot_settings.assert_called_once()

    def test_create_data_managers_only_base(self, simple_manager):
        """Should use base data manager values if no data config."""
        simple_manager.base_data_manager = DataManager(
            n_splits=7, test_size=0.4
        )

        data_managers = simple_manager.create_data_managers()
        data_manager = data_managers["test_group"]
        assert data_manager.test_size == 0.4
        assert data_manager.n_splits == 7

    def test_create_data_managers_one_preprocessor(
        self,
        tmp_path,
        mock_services
    ):
        data_config = {"preprocessors":[ScalingPreprocessor()]}
        with ProjectRootContext(tmp_path):
            experiment_groups = [ExperimentGroupFactory.with_data_config(
                tmp_path, data_config
            )]
            manager = ConfigurationManager(experiment_groups, {})
        manager.set_services(PlotSettings(), mock_services)
        manager.base_data_manager = DataManager()
        data_managers = manager.create_data_managers()
        data_manager = data_managers["test_group"]
        assert data_manager.preprocessors == data_config["preprocessors"]

    def test_create_data_managers_two_preprocessors(
        self,
        tmp_path,
        mock_services
    ):
        data_config = {
            "preprocessors":[ScalingPreprocessor(), MissingDataPreprocessor()],
            "n_splits": 2
        }
        with ProjectRootContext(tmp_path):
            experiment_groups = [ExperimentGroupFactory.with_data_config(
                tmp_path, data_config
            )]
            manager = ConfigurationManager(experiment_groups, {})
        manager.set_services(PlotSettings(), mock_services)
        manager.base_data_manager = DataManager()

        data_managers = manager.create_data_managers()
        data_manager = data_managers["test_group"]
        assert data_manager.preprocessors == data_config["preprocessors"]
        assert data_manager.n_splits == data_config["n_splits"]

    def test_create_data_managers_with_data_config(
        self,
        tmp_path,
        mock_services
    ):
        with ProjectRootContext(tmp_path):
            experiment_groups = [ExperimentGroupFactory.with_data_config(
                tmp_path, {"test_size": 0.6, "split_method": "kfold"}
            )]
            manager = ConfigurationManager(experiment_groups, {})
        manager.set_services(PlotSettings(), mock_services)
        manager.base_data_manager = DataManager(n_splits=7)

        data_managers = manager.create_data_managers()
        data_manager = data_managers["test_group"]
        assert data_manager.test_size == 0.6
        assert data_manager.split_method == "kfold"
        assert data_manager.n_splits == 7

    def test_create_data_managers_data_config_overrides_base_manager(
        self,
        tmp_path,
        mock_services
    ):
        """If base data manager and data_config define same argument the data
        config value overrides the base value."""
        with ProjectRootContext(tmp_path):
            experiment_groups = [ExperimentGroupFactory.with_data_config(
                tmp_path, {"n_splits": 10}
            )]
            manager = ConfigurationManager(experiment_groups, {})
        manager.set_services(PlotSettings(), mock_services)
        manager.base_data_manager = DataManager(n_splits=7)

        data_managers = manager.create_data_managers()
        data_manager = data_managers["test_group"]
        assert data_manager.n_splits == 10

    def test_create_data_managers_add_data_manager_calls(
        self,
        tmp_path,
        mock_services
    ):
        with ProjectRootContext(tmp_path):
            experiment_groups = [
                ExperimentGroupFactory.simple(tmp_path),
                ExperimentGroupFactory.simple(tmp_path, name="group2")
            ]
            manager = ConfigurationManager(experiment_groups, {})
        manager.set_services(PlotSettings(), mock_services)
        manager.base_data_manager = DataManager()

        data_managers = manager.create_data_managers()
        assert manager.services.reporting.add_data_manager.call_count == 2

    def test_create_data_managers_no_base_data_manager(
        self,
        tmp_path,
        mock_services
    ):
        with ProjectRootContext(tmp_path):
            experiment_groups = [ExperimentGroupFactory.simple(tmp_path)]
            manager = ConfigurationManager(experiment_groups, {})
        manager.set_services(PlotSettings(), mock_services)

        with pytest.raises(ValueError):
            data_managers = manager.create_data_managers()

    def test_create_data_splits_with_table_name(self, tmp_path, mock_services):
        categorical_features = {("data.db", "data_table"): ["country"]}
        with ProjectRootContext(tmp_path):
            experiment_groups = [ExperimentGroupFactory.simple(
                tmp_path, datasets=[("data.db", "data_table")]
            )]
            manager = ConfigurationManager(
                experiment_groups, categorical_features
            )
            manager.set_services(PlotSettings(), mock_services)

            mock_data_manager = mock.MagicMock()
            manager.data_managers = {"test_group": mock_data_manager}

            manager.create_data_splits()

        mock_data_manager.split.assert_called_once_with(
            data_path=str(tmp_path / "datasets" / "data.db"),
            categorical_features=["country"],
            group_name="test_group",
            table_name="data_table",
            filename="data"
        )

    def test_create_data_splits_without_table_name(
        self,
        tmp_path,
        mock_services
    ):
        categorical_features = {"data.csv": ["country"]}
        with ProjectRootContext(tmp_path):
            experiment_groups = [ExperimentGroupFactory.simple(
                tmp_path, datasets=["data.csv"]
            )]
            manager = ConfigurationManager(
                experiment_groups, categorical_features
            )
            manager.set_services(PlotSettings(), mock_services)

            mock_data_manager = mock.MagicMock()
            manager.data_managers = {"test_group": mock_data_manager}

            manager.create_data_splits()

        mock_data_manager.split.assert_called_once_with(
            data_path=str(tmp_path / "datasets" / "data.csv"),
            categorical_features=["country"],
            group_name="test_group",
            table_name=None,
            filename="data"
        )

    def test_get_output_structure_with_table_name(
        self,
        tmp_path,
        mock_services
    ):
        with ProjectRootContext(tmp_path):
            experiment_groups = [ExperimentGroupFactory.simple(
                tmp_path, datasets=[("data.db", "data_table")]
            )]
            manager = ConfigurationManager(
                experiment_groups, {}
            )
            manager.set_services(PlotSettings(), mock_services)

            assert manager.get_output_structure() == {
                "test_group": {
                        "data_data_table": (
                        str(tmp_path / "datasets" / "data.db"), "data_table"
                    )
                }
            }

    def test_get_output_structure_without_table_name(
        self,
        tmp_path,
        mock_services
    ):
        with ProjectRootContext(tmp_path):
            experiment_groups = [ExperimentGroupFactory.simple(
                tmp_path, datasets=["data.csv"]
            )]
            manager = ConfigurationManager(
                experiment_groups, {}
            )
            manager.set_services(PlotSettings(), mock_services)

            assert manager.get_output_structure() == {
                "test_group": {
                        "data": (
                        str(tmp_path / "datasets" / "data.csv"), None
                    )
                }
            }

    def test_create_description_map_one_description(self, tmp_path):
        description_map = {"test_group": "This is a test description."}
        with ProjectRootContext(tmp_path):
            experiment_groups = [ExperimentGroupFactory.with_description(
                tmp_path, description_map["test_group"]
            )]
            manager = ConfigurationManager(experiment_groups, {})

        assert description_map == manager.create_description_map()

    def test_create_description_map_no_description(self, tmp_path):
        with ProjectRootContext(tmp_path):
            experiment_groups = [ExperimentGroupFactory.simple(tmp_path)]
            manager = ConfigurationManager(experiment_groups, {})

        assert manager.create_description_map() == {}

    def test_create_description_two_groups(self, tmp_path):
        description_map = {
            "test_group": "This is a test description.",
            "group2": "And this is different description."
        }
        with ProjectRootContext(tmp_path):
            experiment_groups = [
                ExperimentGroupFactory.with_description(
                    tmp_path, description_map["test_group"]
                ),
                ExperimentGroupFactory.with_description(
                    tmp_path, description_map["group2"], name="group2"
                )
            ]
            manager = ConfigurationManager(experiment_groups, {})

        assert description_map == manager.create_description_map()
