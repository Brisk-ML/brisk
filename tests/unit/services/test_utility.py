"""Unit tests for UtilityService."""

import pytest
import numpy as np
import pandas as pd
import sklearn.model_selection as model_select
from unittest import mock

from brisk.services import utility


class GroupIndexFactory:
    @classmethod
    def simple(cls, size: int = 100):
        return {
            "indices": np.random.randint(0, 10, size=size)
        }
    
    @classmethod
    def train_test_pair(cls, train_size: int = 100, test_size: int = 30):
        return (
            cls.simple(train_size),
            cls.simple(test_size)
        )


@pytest.fixture
def group_index_train():
    return GroupIndexFactory.simple(size=100)


@pytest.fixture
def group_index_test():
    return GroupIndexFactory.simple(size=30)


@pytest.fixture
def utility_service_no_groups():
    return utility.UtilityService("test_utility", None, None)


@pytest.fixture
def utility_service_with_groups(group_index_train, group_index_test):
    return utility.UtilityService("test_utility", group_index_train, group_index_test)


@pytest.fixture
def mock_logging_service():
    logging_service = mock.MagicMock()
    logging_service.logger = mock.MagicMock()
    return logging_service


@pytest.fixture
def categorical_target():
    y = pd.Series([0, 1, 2] * 33 + [0], name="target")
    y.attrs["is_test"] = False
    return y


@pytest.fixture
def continuous_target():
    y = pd.Series(np.random.randn(100), name="target")
    y.attrs["is_test"] = False
    return y


class TestUtilityService:
    def test_set_split_indices_no_indices(self, utility_service_no_groups):
        """Test set_split_indices with no group indices."""
        utility_service_no_groups.set_split_indices(None, None)
        
        assert utility_service_no_groups.group_index_train is None
        assert utility_service_no_groups.group_index_test is None
        assert utility_service_no_groups.data_has_groups is False

    def test_set_split_indices_train_index(self, utility_service_no_groups, group_index_train):
        utility_service_no_groups.set_split_indices(group_index_train, None)
        
        assert utility_service_no_groups.group_index_train is not None
        assert utility_service_no_groups.group_index_test is None
        assert utility_service_no_groups.data_has_groups is False
        
    def test_set_split_indices_test_index(self, utility_service_no_groups, group_index_train):
        utility_service_no_groups.set_split_indices(None, group_index_train)
        
        assert utility_service_no_groups.group_index_train is None
        assert utility_service_no_groups.group_index_test is not None
        assert utility_service_no_groups.data_has_groups is False

    def test_set_split_indices_both_indices(
        self, utility_service_no_groups, group_index_train, group_index_test
    ):
        """Test set_split_indices with both group indices."""
        utility_service_no_groups.set_split_indices(group_index_train, group_index_test)
        
        assert utility_service_no_groups.group_index_train is not None
        assert utility_service_no_groups.group_index_test is not None
        assert utility_service_no_groups.data_has_groups is True

    def test_get_group_index_no_groups(self, utility_service_no_groups):
        """Test get_group_index when data has no groups."""
        train_index = utility_service_no_groups.get_group_index(is_test=False)
        test_index = utility_service_no_groups.get_group_index(is_test=True)
        
        assert train_index is None
        assert test_index is None

    def test_get_group_index_train_group(self, utility_service_with_groups, group_index_train):
        """Test get_group_index for training data."""
        train_index = utility_service_with_groups.get_group_index(is_test=False)
        
        assert train_index == group_index_train

    def test_get_group_index_test_group(self, utility_service_with_groups, group_index_test):
        """Test get_group_index for test data."""
        test_index = utility_service_with_groups.get_group_index(is_test=True)
        
        assert test_index == group_index_test

    def test_get_cv_splitter_group_categorical_num_repeats(
        self, utility_service_with_groups, categorical_target, mock_logging_service
    ):
        """Test get_cv_splitter with grouped categorical data and num_repeats.
        
        Should use StratifiedGroupKFold and log warning.
        """
        utility_service_with_groups._other_services = {"logging": mock_logging_service}
        
        splitter, indices = utility_service_with_groups.get_cv_splitter(
            categorical_target, cv=5, num_repeats=3
        )
        
        assert isinstance(splitter, model_select.StratifiedGroupKFold)
        mock_logging_service.logger.warning.assert_called_once()

    def test_get_cv_splitter_group_non_categorical_num_repeats(
        self, utility_service_with_groups, continuous_target, mock_logging_service
    ):
        """Test get_cv_splitter with grouped continuous data and num_repeats.
        
        Should use GroupKFold and log warning.
        """
        utility_service_with_groups._other_services = {"logging": mock_logging_service}
        
        splitter, indices = utility_service_with_groups.get_cv_splitter(
            continuous_target, cv=5, num_repeats=3
        )
        
        assert isinstance(splitter, model_select.GroupKFold)
        mock_logging_service.logger.warning.assert_called_once()

    def test_get_cv_splitter_group_categorical(
        self, utility_service_with_groups, categorical_target
    ):
        """Test get_cv_splitter with grouped categorical data.
        
        Should use StratifiedGroupKFold.
        """
        splitter, indices = utility_service_with_groups.get_cv_splitter(
            categorical_target, cv=5
        )
        
        assert isinstance(splitter, model_select.StratifiedGroupKFold)
        assert len(indices) == len(categorical_target)

    def test_get_cv_splitter_group_non_categorical(
        self, utility_service_with_groups, continuous_target
    ):
        """Test get_cv_splitter with grouped continuous data.
        
        Should use GroupKFold.
        """
        splitter, indices = utility_service_with_groups.get_cv_splitter(
            continuous_target, cv=5
        )
        
        assert isinstance(splitter, model_select.GroupKFold)

    def test_get_cv_splitter_categorical_num_repeats(
        self, utility_service_no_groups, categorical_target
    ):
        """Test get_cv_splitter with categorical data and num_repeats.
        
        Should use RepeatedStratifiedKFold.
        """
        splitter, indices = utility_service_no_groups.get_cv_splitter(
            categorical_target, cv=5, num_repeats=3
        )
        
        assert isinstance(splitter, model_select.RepeatedStratifiedKFold)

    def test_get_cv_splitter_non_categorical_num_repeats(
        self, utility_service_no_groups, continuous_target
    ):
        """Test get_cv_splitter with continuous data and num_repeats.
        
        Should use RepeatedKFold.
        """
        splitter, indices = utility_service_no_groups.get_cv_splitter(
            continuous_target, cv=5, num_repeats=3
        )
        
        assert isinstance(splitter, model_select.RepeatedKFold)

    def test_get_cv_splitter_categorical(self, utility_service_no_groups, categorical_target):
        """Test get_cv_splitter with categorical data.
        
        Should use StratifiedKFold.
        """
        splitter, indices = utility_service_no_groups.get_cv_splitter(
            categorical_target, cv=5
        )
        
        assert isinstance(splitter, model_select.StratifiedKFold)

    def test_get_cv_splitter_non_categorical(self, utility_service_no_groups, continuous_target):
        """Test get_cv_splitter with continuous data.
        
        Should use KFold.
        """
        splitter, indices = utility_service_no_groups.get_cv_splitter(
            continuous_target, cv=5
        )
        
        assert isinstance(splitter, model_select.KFold)

    def test_get_cv_splitter_group_has_indices(
        self, utility_service_with_groups, categorical_target, group_index_train
    ):
        """Test that get_cv_splitter returns correct indices for grouped data."""
        splitter, indices = utility_service_with_groups.get_cv_splitter(
            categorical_target, cv=5
        )
        
        assert indices is not None
        assert np.array_equal(indices, group_index_train["indices"])
