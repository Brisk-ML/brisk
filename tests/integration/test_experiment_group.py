"""Integration tests for ExperimentGroup."""
    #
    # def test_missing_dataset(self, mock_brisk_project):
    #     """Test creation with non-existent dataset"""
    #     with pytest.raises(FileNotFoundError, match="Dataset not found"):
    #         ExperimentGroup(
    #             name="test",
    #             workflow="regression_workflow",
    #             datasets=["nonexistent.csv"]
    #         )
    #

    # def test_dataset_paths(self, valid_group_two_datasets, mock_brisk_project):
    #     """Test dataset_paths property"""
    #     expected_paths = [
    #         mock_brisk_project / 'datasets' / 'regression.csv',
    #         mock_brisk_project / 'datasets' / 'categorical.csv'
    #     ]
    #     actual_paths  = [
    #         path for path, _ in valid_group_two_datasets.dataset_paths
    #     ]
    #     assert actual_paths == expected_paths

    # def test_invalid_algorithm_config(self, mock_brisk_project):
    #     """Test creation with invalid algorithm configuration"""
    #     with pytest.raises(
    #         ValueError, 
    #         match="Algorithm config contains algorithms not in the list of algorithms:"
    #     ):
    #         ExperimentGroup(
    #             name="test",
    #             workflow="regression_workflow",
    #             datasets=["regression.csv"],
    #             algorithms=["linear"],
    #             algorithm_config={"ridge": {"alpha": 1.0}}
    #         )
    #

    # def test_invalid_algorithm_config_nested(self, mock_brisk_project):
    #     """Test creation with invalid algorithm configuration"""
    #     with pytest.raises(
    #         ValueError, 
    #         match="Algorithm config contains algorithms not in the list of algorithms:"
    #     ):
    #         ExperimentGroup(
    #             name="test",
    #             workflow="regression_workflow",
    #             datasets=["regression.csv"],
    #             algorithms=[["linear", "ridge"]],
    #             algorithm_config={"elasticnet": {"alpha": 1.0}}
    #         )
    #
    #     # Check nested alorithms are found correctly
    #     ExperimentGroup(
    #         name="test",
    #         workflow="regression_workflow",
    #         datasets=["regression.csv"],
    #         algorithms=[["linear", "ridge"]],
    #         algorithm_config={"ridge": {"alpha": 1.0}}
    #     )
    #
