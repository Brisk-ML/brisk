"""Integration tests for ConfigurationManager"""
from unittest import mock

import pytest

from brisk.configuration import project, configuration_manager
from brisk.theme import plot_settings
from brisk.data import data_manager
from brisk import services

from tests.utils.factories import ExperimentGroupFactory, AlgorithmFactory


class TestConfigurationManagerIntegration:
    def test_create_experiment_queue_missing_workflow(self, tmp_path):
        with project.ProjectRootContext(tmp_path):
            experiment_groups = [ExperimentGroupFactory.simple(
                tmp_path,
                workflow="does_not_exist"
            )]
            manager = configuration_manager.ConfigurationManager(
                experiment_groups,
                {},
            ) 
            services.initialize_services(tmp_path)
            manager.set_services(plot_settings.PlotSettings())
            manager.algorithm_config = AlgorithmFactory.collection()
            manager.base_data_manager = data_manager.DataManager()

            with pytest.raises(ImportError):
                manager.create_experiment_queue()

    def test_create_experiment_queue_0_n_splits(self, tmp_path):
        with project.ProjectRootContext(tmp_path):
            experiment_groups = [ExperimentGroupFactory.simple(
                tmp_path,
                workflow="does_not_exist"
            )]
            manager = configuration_manager.ConfigurationManager(
                experiment_groups,
                {},
            ) 
            services.initialize_services(tmp_path)
            manager.set_services(plot_settings.PlotSettings(), mock.Mock())
            manager.algorithm_config = AlgorithmFactory.collection()
            manager.base_data_manager = data_manager.DataManager(n_splits=0)

            queue = manager.create_experiment_queue()
            assert len(queue) == 0

    def test_create_experiment_queue_1_n_splits(self, tmp_path):
        with project.ProjectRootContext(tmp_path):
            experiment_groups = [ExperimentGroupFactory.simple(
                tmp_path,
                workflow="does_not_exist"
            )]
            manager = configuration_manager.ConfigurationManager(
                experiment_groups,
                {},
            ) 
            services.initialize_services(tmp_path)
            manager.set_services(plot_settings.PlotSettings(), mock.Mock())
            manager.algorithm_config = AlgorithmFactory.collection()
            manager.base_data_manager = data_manager.DataManager(n_splits=1)

            queue = manager.create_experiment_queue()
            assert len(queue) == 1

    def test_create_experiment_queue_2_n_splits(self, tmp_path):
        with project.ProjectRootContext(tmp_path):
            experiment_groups = [ExperimentGroupFactory.simple(
                tmp_path,
                workflow="does_not_exist"
            )]
            manager = configuration_manager.ConfigurationManager(
                experiment_groups,
                {},
            ) 
            services.initialize_services(tmp_path)
            manager.set_services(plot_settings.PlotSettings(), mock.Mock())
            manager.algorithm_config = AlgorithmFactory.collection()
            manager.base_data_manager = data_manager.DataManager(n_splits=2)

            queue = manager.create_experiment_queue()
            assert len(queue) == 2
