"""Integration tests for Configuration"""

# NOTE: This test requires the services layer, Configuration and 
# ConfigurationManager communicate. There are also several other classes
# involved making isolation difficult.
#
# @patch("brisk.configuration.experiment_group.ExperimentGroup._validate_datasets")
# def test_build_returns_configuration_manager(
#         self, mock_validate_datasets
# ):
#     """Test build method returns ConfigurationManager"""
#     mock_validate_datasets.return_value = None
#     configuration = ConfigurationFactory.simple()
#     configuration.add_experiment_group(
#         name="test_group",
#         datasets=["regression.csv"]
#     )
#
#     manager = configuration.build()
#     assert isinstance(manager, ConfigurationManager)
#     assert manager.experiment_groups == configuration.experiment_groups
#     assert manager.categorical_features == configuration.categorical_features
#
