"""Unit tests for DataSplitInfo."""
from unittest.mock import MagicMock, Mock, patch

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.preprocessing import MinMaxScaler

from brisk.data.data_split_info import DataSplitInfo
from brisk.data.splitkey import SplitKey

from tests.utils.factories import DataFrameFactory

@pytest.fixture
def split_key():
   return SplitKey(
        group_name="test_group",
        dataset_name="data",
        table_name=None
    )


@pytest.mark.unit
class TestDataSplitInfoUnit:
    def test_initialization(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["continuous", "categorical"]
        )
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1,
            scaler = MinMaxScaler(),
            categorical_features=["feature_0"],
            continuous_features=["feature_1"]
        )
        assert split.features.sort() == ["feature_0", "feature_1"].sort()
        assert split.dataset_name == split_key.dataset_name

    def test_initializatioin_no_optional(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["continuous", "categorical"]
        )
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1
        )
        assert split.features.sort() == ["feature_0", "feature_1"].sort()
        assert split.dataset_name == split_key.dataset_name

    def test_init_empty_dataframe(self, split_key):
        data = {
            "X_train": pd.DataFrame(),
            "X_test": pd.DataFrame(),
            "y_train": pd.Series(),
            "y_test": pd.Series(),
            "group_index_train": {},
            "group_index_test": {}
        }
        with pytest.raises(ValueError):
            split = DataSplitInfo(
                **data,
                split_key = split_key,
                split_index = 1
            )

    def test_init_creates_deep_copy(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["continuous", "categorical"]
        )
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1
        )
        data["X_train"].loc[0, "feature_0"] = 12345
        assert split.X_train.loc[0, "feature_0"] != 12345

        data["y_train"].loc[0] = 12345
        assert split.y_train.loc[0] != 12345

        data["X_test"].loc[0, "feature_0"] = 12345
        assert split.X_test.loc[0, "feature_0"] != 12345

        data["y_test"].loc[0] = 12345
        assert split.y_test.loc[0] != 12345

    def test_evaluate_data_split_clear_context_happy(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["continuous", "categorical"]
        )
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1
        )
        mock_services = MagicMock()
        split.set_services(mock_services)

        # Use mocks instead of real evaluators
        with patch.object(split.registry, "get", return_value=MagicMock()):
            split.evaluate_data_split()

        mock_services.reporting.set_context.assert_called_once()
        mock_services.reporting.clear_context.assert_called_once()

    def test_evaluate_data_split_clear_context_error(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["continuous", "categorical"]
        )
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1
        )
        mock_services = MagicMock()
        split.set_services(mock_services)

        # Use mocks instead of real evaluators
        failing_evaluator = MagicMock()
        failing_evaluator.evaluate.side_effect = Exception("A mock error")
        with patch.object(
            split.registry, "get", return_value=failing_evaluator
        ):
            with pytest.raises(Exception):
                split.evaluate_data_split()

        mock_services.reporting.set_context.assert_called_once()
        mock_services.reporting.clear_context.assert_called_once()

    def test_detect_categorical_features_1_percent_unique(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=100, n_features=1, problem_type="regression",
            feature_types=["categorical"]
        )
        data["X_train"] = pd.DataFrame({
            "feature_0": [0] * 80,
        })
        data["X_test"] = pd.DataFrame({
            "feature_0": [0] * 20
        })
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1
        )
        assert split.categorical_features == ["feature_0"]

    def test_detect_categorical_features_5_percent_unique(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=100, n_features=1, problem_type="regression",
            feature_types=["categorical"]
        )
        data["X_train"] = pd.DataFrame({
            "feature_0": [1, 2, 3, 4] * 20,
        })
        data["X_test"] = pd.DataFrame({
            "feature_0": [0] * 20
        })
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1
        )
        assert split.categorical_features == ["feature_0"]

    def test_detect_categorical_features_10_percent_unique(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=100, n_features=1, problem_type="regression",
            feature_types=["categorical"]
        )
        data["X_train"] = pd.DataFrame({
            "feature_0": [num for num in range(1, 96)],
        })
        data["X_test"] = pd.DataFrame({
            "feature_0": [0] * 20
        })
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1
        )
        assert split.categorical_features == []

    def test_detect_categorical_features_string_data(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=100, n_features=2, problem_type="regression",
            feature_types=["categorical"] * 2
        )
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1
        )
        assert split.categorical_features == ["feature_0", "feature_1"]

    def test_detect_categorical_features_pd_categorical_data(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=100, n_features=1, problem_type="regression",
            feature_types=["categorical"]
        )
        data["X_train"] = pd.DataFrame({
            "feature_0": pd.Series(["cat", "dog", "bird"], dtype="category")
        })
        data["X_test"] = pd.DataFrame({
            "feature_0": pd.Series(["cat", "dog", "bird"], dtype="category")
        })
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1
        )
        assert split.categorical_features == ["feature_0"]

    def test_get_train_scaler_fitted(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=1, problem_type="regression",
            feature_types=["continuous"]
        )
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.zeros((len(data["X_train"]), 1))
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1,
            scaler=mock_scaler,
            continuous_features=["feature_0"]
        )
        x_train, y_train = split.get_train()
        mock_scaler.transform.assert_called_once()
        assert_series_equal(y_train, data["y_train"])
        assert x_train.size == data["X_train"].size

    def test_get_train_no_scaler(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["continuous", "categorical"]
        )
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1,
            continuous_features=["feature_0"]
        )
        x_train, y_train = split.get_train()
        assert_series_equal(y_train, data["y_train"])
        assert_frame_equal(x_train, data["X_train"])
        assert x_train.size == data["X_train"].size

    def test_get_train_categorical_and_scaler(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["continuous", "categorical"]
        )
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.zeros((len(data["X_train"]), 1))
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1,
            scaler=mock_scaler,
            continuous_features=["feature_0"]
        )
        x_train, y_train = split.get_train()
        mock_scaler.transform.assert_called_once()
        assert_series_equal(y_train, data["y_train"])
        assert x_train.size == data["X_train"].size

    def test_get_train_correct_column_order(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=6, problem_type="regression",
            feature_types=["continuous", "categorical", "continuous"] * 2
        )
        column_order = data["X_train"].columns
        minmax_scaler = MinMaxScaler().fit(data["X_train"].iloc[:, [0,2,3,5]])
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1,
            scaler=minmax_scaler,
        )
        x_train, _ = split.get_train()
        assert x_train.size == data["X_train"].size
        assert x_train.columns.all() == column_order.all()

    def test_get_train_correct_index(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["continuous", "categorical"]
        )
        minmax_scaler = MinMaxScaler().fit(data["X_train"].iloc[:, [0]])
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1,
            scaler=minmax_scaler,
        )
        x_train, _ = split.get_train()
        assert x_train.size == data["X_train"].size
        assert data["X_train"].index.equals(x_train.index)

    def test_get_test_scaler_fitted(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=1, problem_type="regression",
            feature_types=["continuous"]
        )
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.zeros((len(data["X_test"]), 1))
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1,
            scaler=mock_scaler,
            continuous_features=["feature_0"]
        )
        x_test, y_test = split.get_test()
        mock_scaler.transform.assert_called_once()
        assert_series_equal(y_test, data["y_test"])
        assert x_test.size == data["X_test"].size

    def test_get_test_no_scaler(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["continuous", "categorical"]
        )
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1,
            continuous_features=["feature_0"]
        )
        x_test, y_test = split.get_test()
        assert_series_equal(y_test, data["y_test"])
        assert_frame_equal(x_test, data["X_test"])
        assert x_test.size == data["X_test"].size

    def test_get_test_categorical_and_scaler(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["continuous", "categorical"]
        )
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.zeros((len(data["X_test"]), 1))
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1,
            scaler=mock_scaler,
            continuous_features=["feature_0"]
        )
        x_test, y_test = split.get_test()
        mock_scaler.transform.assert_called_once()
        assert_series_equal(y_test, data["y_test"])
        assert x_test.size == data["X_test"].size

    def test_get_test_correct_column_order(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=6, problem_type="regression",
            feature_types=["continuous", "categorical", "continuous"] * 2
        )
        column_order = data["X_test"].columns
        minmax_scaler = MinMaxScaler().fit(data["X_test"].iloc[:, [0,2,3,5]])
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1,
            scaler=minmax_scaler,
        )
        x_test, _ = split.get_test()
        assert x_test.size == data["X_test"].size
        assert x_test.columns.all() == column_order.all()

    def test_get_test_correct_index(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["continuous", "categorical"]
        )
        minmax_scaler = MinMaxScaler().fit(data["X_test"].iloc[:, [0]])
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1,
            scaler=minmax_scaler,
        )
        x_test, _ = split.get_test()
        assert x_test.size == data["X_test"].size
        assert data["X_test"].index.equals(x_test.index)

    def test_get_train_test_order(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["continuous", "categorical"]
        )
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1
        )
        train_test = split.get_train_test()
        assert_frame_equal(data["X_train"], train_test[0])
        assert_frame_equal(data["X_test"], train_test[1])
        assert_series_equal(data["y_train"], train_test[2])
        assert_series_equal(data["y_test"], train_test[3])

    def test_get_split_metadata_categorical_features(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=22, n_features=3, problem_type="regression",
            feature_types=["categorical"] * 3
        )
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1
        )
        split_metadata = split.get_split_metadata()
        assert split_metadata["num_features"] == 3
        assert split_metadata["num_samples"] == 22

    def test_get_split_metadata_continuous_features(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=18, n_features=5, problem_type="regression",
            feature_types=["continuous"] * 5
        )
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1
        )
        split_metadata = split.get_split_metadata()
        assert split_metadata["num_features"] == 5
        assert split_metadata["num_samples"] == 18

    def test_get_split_metadata_mixed_features(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=11, n_features=4, problem_type="regression",
            feature_types=["continuous", "categorical"] * 2
        )
        split = DataSplitInfo(
            **data,
            split_key = split_key,
            split_index = 1
        )
        split_metadata = split.get_split_metadata()
        assert split_metadata["num_features"] == 4
        assert split_metadata["num_samples"] == 11

    def test_set_features(self, split_key):
        """Test _set_features with both categorical and continuous provided."""
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=4, problem_type="regression",
            feature_types=[
                "continuous", "continuous", "categorical", "categorical"
            ]
        )
        split = DataSplitInfo(
            **data,
            split_key=split_key,
            split_index=0,
            categorical_features=["feature_2", "feature_3"],
            continuous_features=["feature_0", "feature_1"]
        )
        assert sorted(split.categorical_features) == ["feature_2", "feature_3"]
        assert sorted(split.continuous_features) == ["feature_0", "feature_1"]
        assert sorted(split.features) == [
            "feature_0", "feature_1", "feature_2", "feature_3"
        ]

    def test_set_features_no_optional(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["continuous", "categorical"]
        )
        split = DataSplitInfo(
            **data,
            split_key=split_key,
            split_index=0,
            categorical_features=None,
            continuous_features=None
        )
        assert sorted(split.categorical_features) == ["feature_1"]
        assert sorted(split.continuous_features) == ["feature_0"]
        assert sorted(split.features) == ["feature_0", "feature_1"]

    def test_set_features_detect_none_categorical(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["continuous", "categorical"]
        )
        split = DataSplitInfo(
            **data,
            split_key=split_key,
            split_index=0,
            categorical_features=None,
            continuous_features=["feature_0"]
        )
        assert sorted(split.categorical_features) == ["feature_1"]
        assert sorted(split.continuous_features) == ["feature_0"]
        assert sorted(split.features) == ["feature_0", "feature_1"]

    def test_set_features_detect_len_0_categorical(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["categorical", "continuous"]
        )
        split = DataSplitInfo(
            **data,
            split_key=split_key,
            split_index=0,
            categorical_features=[],
            continuous_features=["feature_1"]
        )
        assert sorted(split.categorical_features) == ["feature_0"]
        assert sorted(split.continuous_features) == ["feature_1"]
        assert sorted(split.features) == ["feature_0", "feature_1"]

    def test_set_features_categorical_features_not_in_columns(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["continuous", "categorical"]
        )
        split = DataSplitInfo(
            **data,
            split_key=split_key,
            split_index=0,
            categorical_features=["feature_1", "nonexistent_feature"],
            continuous_features=["feature_0"]
        )
        assert sorted(split.categorical_features) == ["feature_1"]
        assert sorted(split.continuous_features) == ["feature_0"]
        assert sorted(split.features) == ["feature_0", "feature_1"]

    def test_set_features_detect_none_continuous(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["continuous", "categorical"]
        )
        split = DataSplitInfo(
            **data,
            split_key=split_key,
            split_index=0,
            categorical_features=["feature_1"],
            continuous_features=None
        )
        assert sorted(split.categorical_features) == ["feature_1"]
        assert sorted(split.continuous_features) == ["feature_0"]
        assert sorted(split.features) == ["feature_0", "feature_1"]

    def test_set_features_detect_len_0_continuous(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["continuous", "categorical"]
        )
        split = DataSplitInfo(
            **data,
            split_key=split_key,
            split_index=0,
            categorical_features=["feature_1"],
            continuous_features=[]
        )
        assert sorted(split.categorical_features) == ["feature_1"]
        assert sorted(split.continuous_features) == ["feature_0"]
        assert sorted(split.features) == ["feature_0", "feature_1"]

    def test_set_features_continuous_excludes_categorical(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=3, problem_type="regression",
            feature_types=["continuous", "continuous", "categorical"]
        )
        split = DataSplitInfo(
            **data,
            split_key=split_key,
            split_index=0,
            categorical_features=["feature_2"],
            continuous_features=["feature_0", "feature_1", "feature_2"]
        )
        assert sorted(split.categorical_features) == ["feature_2"]
        assert sorted(split.continuous_features) == ["feature_0", "feature_1"]
        assert sorted(split.features) == ["feature_0", "feature_1", "feature_2"]

    def test_set_features_continuous_not_in_columns(self, split_key):
        data = DataFrameFactory.train_test_split(
            n_samples=10, n_features=2, problem_type="regression",
            feature_types=["continuous", "categorical"]
        )
        split = DataSplitInfo(
            **data,
            split_key=split_key,
            split_index=0,
            categorical_features=["feature_1"],
            continuous_features=["feature_0", "nonexistent_continuous"]
        )
        assert sorted(split.categorical_features) == ["feature_1"]
        assert sorted(split.continuous_features) == ["feature_0"]
        assert sorted(split.features) == ["feature_0", "feature_1"]
