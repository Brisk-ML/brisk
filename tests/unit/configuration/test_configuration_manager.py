"""Unit tests for ConfigurationManager"""
from unittest.mock import patch
import textwrap

import pytest

from brisk import ConfigurationManager, DataManager
from brisk.theme.plot_settings import PlotSettings

from tests.utils.factories import (
    ExperimentGroupFactory, DataManagerFactory, AlgorithmFactory
)
from tests.utils.mocks import MockServiceBundle

# pylint: disable=W0621, W0613

@pytest.fixture
def base_data_manager():
    return DataManagerFactory.full(
        test_size=0.2,
        n_splits=2,
        split_method="shuffle",
        group_column=None,
        stratified=False,
        random_state=42,
        problem_type="classification",
        algorithm_config=None,
        preprocessors=[]
    )


class TestConfigurationManager:
    """Unit tests for the ConfigurationManager class."""
    @patch("brisk.configuration.configuration_manager.get_services")
    def test_get_base_params(
        self,
        mock_services,
        base_data_manager
    ):
        """Test the _get_base_params method of ConfigurationManager."""
        mock_services.return_value = MockServiceBundle()
        group = ExperimentGroupFactory.simple(
            datasets=["regression.csv"],
            algorithms=["linear"]
        )
        manager = ConfigurationManager([group], {}, PlotSettings())
        manager.base_data_manager = base_data_manager
        base_params = manager._get_base_params()
        expected_params = {
            "test_size": 0.2,
            "n_splits": 2,
            "split_method": "shuffle",
            "group_column": None,
            "stratified": False,
            "random_state": 42,
            "problem_type": "classification",
            "algorithm_config": None,
            "preprocessors": []
        }
        assert base_params == expected_params

    @patch("brisk.configuration.configuration_manager.get_services")
    def test_data_config_does_not_change_base(
        self,
        mock_services,
        base_data_manager
    ):
        """Test passing data_config arg does not change the base data manager"""
        mock_services.return_value = MockServiceBundle()
        group = ExperimentGroupFactory.simple()
        group.data_config = {
            "test_size": 0.4,
            "split_method": "kfold"
            }
        manager = ConfigurationManager([group], {}, PlotSettings())
        manager.base_data_manager = base_data_manager
        base_params = manager._get_base_params()
        expected_params = {
            "test_size": 0.2,
            "n_splits": 2,
            "split_method": "shuffle",
            "group_column": None,
            "stratified": False,
            "random_state": 42,
            "problem_type": "classification",
            "algorithm_config": None,
            "preprocessors": []
        }
        assert base_params == expected_params

    @patch("brisk.configuration.configuration_manager.get_services")
    @patch("brisk.data.data_manager.get_services")
    def test_data_manager_reuse(
        self,
        mock_data_services,
        mock_services,
        base_data_manager
    ):
        """Test that DataManagers are reused for matching configurations."""
        mock_services.return_value = MockServiceBundle()
        groups = [
            ExperimentGroupFactory.simple(
                name="group1",
                workflow="regression_workflow",
                datasets=["regression.csv"],
                algorithms=["linear"]
            ),
            ExperimentGroupFactory.simple(
                name="group2",
                workflow="regression_workflow",
                datasets=["categorical.csv"],
                algorithms=["ridge"]
            ),
            ExperimentGroupFactory.simple(
                name="group3",
                workflow="regression_workflow",
                datasets=["regression.csv"],
                algorithms=["elasticnet"]
            )
        ]
        groups[1].data_config = {"test_size": 0.3}
        manager = ConfigurationManager(
            groups, {"categorical.csv": ["category"]}, PlotSettings()
        )
        manager.base_data_manager = base_data_manager
        manager.get_data_managers()
        for data_manager in manager.data_managers.values():
            assert isinstance(data_manager, DataManager)
        assert len(manager.data_managers) == 3
        # group1 and group3 should share the base DataManager
        assert (
            manager.data_managers["group1"] is manager.data_managers["group3"]
        )
        # group2 should have its own DataManager
        assert (
            manager.data_managers["group2"]
            is not manager.data_managers["group1"]
        )

    def test_experiment_creation(self, base_data_manager):
        """Test creation of experiments from groups."""
        groups = [
            ExperimentGroupFactory.simple(
                name="single",
                workflow="regression_workflow",
                datasets=["regression.csv"],
            ),
            ExperimentGroupFactory.with_multiple_datasets(2)
        ]
        manager = ConfigurationManager(groups, {}, PlotSettings())
        manager.algorithm_config = AlgorithmFactory.collection()
        manager.base_data_manager = base_data_manager
        experiment_queue = manager.get_experiment_queue()
        assert len(experiment_queue) == 6

    def test_correct_experiment_queue_length(self, base_data_manager):
        """Test the correct length of the experiment queue."""
        groups = [
            ExperimentGroupFactory.simple(
                name="group1",
                workflow="regression_workflow",
                datasets=["regression.csv", "categorical.csv"],
                algorithms=["linear", "ridge", "lasso"]
            ),
            ExperimentGroupFactory.simple(
                name="group2",
                workflow="regression_workflow",
                datasets=["regression.csv"],
                algorithms=[["ridge", "lasso"], ["linear", "ridge"]]
            ),
            ExperimentGroupFactory.simple(
                name="group3",
                workflow="regression_workflow",
                datasets=["regression.csv"],
                algorithms=["linear", "ridge"]
            )
        ]
        manager = ConfigurationManager(groups, {}, PlotSettings())
        manager.algorithm_config = AlgorithmFactory.collection()
        manager.base_data_manager = base_data_manager
        experiment_queue = manager.get_experiment_queue()
        assert len(experiment_queue) == 20

    def test_get_output_structure(self):
        """Test the _get_output_structure method of ConfigurationManager."""
        group1 = ExperimentGroupFactory.simple(
            name="group1",
            workflow="regression_workflow",
            datasets=["regression.csv"],
            algorithms=["linear"],
        )

        group2 = ExperimentGroupFactory.simple(
            name="group2",
            workflow="regression_workflow",
            datasets=["regression.csv"],
            algorithms=["ridge"],
        )

        manager = ConfigurationManager([group1, group2], {}, PlotSettings())
        output_structure = manager.get_output_structure()
        expected_output_structure = {
            "group1": {
                "regression": ("datasets/regression.csv", None)
            },
            "group2": {
                "regression": ("datasets/regression.csv", None)
            }
        }

        assert output_structure == expected_output_structure

    def test_get_output_structure_with_sql(self):
        """Test get_output_structure uses correct SQL table name."""
        group1 = ExperimentGroupFactory.simple(
            name="group1",
            workflow="regression_workflow",
            datasets=[("test_data.db", "regression")],
            algorithms=["linear"],
        )

        manager = ConfigurationManager([group1], {}, PlotSettings())
        output_structure = manager.get_output_structure()

        expected_output_structure = {
            "group1": {
                "test_data_regression": ("datasets/test_data.db", "regression")
            }
        }
        assert output_structure == expected_output_structure

    def test_get_output_structure_with_multiple_datasets(self):
        """Test the get_output_structure with multiple datasets."""
        group1 = ExperimentGroupFactory.simple(
            name="group1",
            workflow="regression_workflow",
            datasets=["regression.csv", "categorical.csv"],
            algorithms=["linear"]
        )
        group2 = ExperimentGroupFactory.simple(
            name="group2",
            workflow="regression_workflow",
            datasets=[
                "regression.csv",
                ("test_data.db", "regression"),
                ("test_data.db", "categorical")
            ],
            algorithms=["ridge"]
        )
        manager = ConfigurationManager([group1, group2], {
            "categorical.csv": ["category"],
            ("test_data.db", "categorical"): ["category", "result"]
        }, PlotSettings())
        output_structure = manager.get_output_structure()
        expected_output_structure = {
            "group1": {
                "regression": ("datasets/regression.csv", None),
                "categorical": ("datasets/categorical.csv", None)
            },
            "group2": {
                "regression": ("datasets/regression.csv", None),
                "test_data_regression": ("datasets/test_data.db", "regression"),
                "test_data_categorical": (
                    "datasets/test_data.db", "categorical"
                )
            }
        }
        assert output_structure == expected_output_structure

    def test_create_description_map(self):
        """Test the _create_description_map method of ConfigurationManager."""
        group1 = ExperimentGroupFactory.simple(
            name="group1",
            workflow="regression_workflow",
            datasets=["regression.csv"],
            algorithms=["linear"]
        )
        group1.description="This is a test description"
        group2 = ExperimentGroupFactory.simple(
            name="group2",
            workflow="regression_workflow",
            datasets=["regression.csv"],
            algorithms=["ridge"]
        )
        group2.description="This is another test description that needs to be wrapped and stored properly." # pylint: disable=C0301

        group3 = ExperimentGroupFactory.simple(
            name="group3",
            workflow="regression_workflow",
            datasets=["regression.csv"],
            algorithms=["elasticnet"]
        )
        manager = ConfigurationManager(
            [group1, group2, group3], {}, PlotSettings()
        )
        expected_group2_description = textwrap.dedent("""
        This is another test description that needs to be wrapped and stored properly.
        """).strip()

        assert manager.get_description_map() == {
            "group1": "This is a test description",
            "group2": expected_group2_description
        }
