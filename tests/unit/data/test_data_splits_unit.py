"""Unit tests for DataSplits."""

import pytest

from brisk.data.data_splits import DataSplits
from brisk.data.data_split_info import DataSplitInfo
from brisk.data.splitkey import SplitKey

from tests.utils.factories import DataFrameFactory


@pytest.fixture
def data_split_info():
    data = DataFrameFactory.train_test_split(
        n_samples=5,
        n_features=1,
        problem_type="regression"
    )
    split_key = SplitKey(
        group_name="test_group",
        dataset_name="data",
        table_name=None
    )
    return DataSplitInfo(
        **data,
        split_key=split_key,
        split_index=1
    )


class TestDataSplitsUnit:
    def test_initalization_0_splits(self):
        with pytest.raises(ValueError):
            splits = DataSplits(0)

    def test_initalization_1_split(self):
        n_splits = 1
        splits = DataSplits(n_splits)
        assert len(splits) == n_splits
        assert len(splits._data_splits) == n_splits

    def test_initalization_2_splits(self):
        n_splits = 2
        splits = DataSplits(n_splits)
        assert len(splits) == n_splits
        assert len(splits._data_splits) == n_splits

    def test_initalization_negative_splits(self):
        with pytest.raises(ValueError):
            splits = DataSplits(-1)

    def test_add_1(self, data_split_info):
        splits = DataSplits(2)
        splits.add(data_split_info)
        assert splits._current_index == 1

    def test_add_2(self, data_split_info):
        splits = DataSplits(2)
        splits.add(data_split_info)
        splits.add(data_split_info)
        assert splits._current_index == 2

    def test_add_more_than_expected(self, data_split_info):
        splits = DataSplits(1)
        splits.add(data_split_info)
        with pytest.raises(IndexError):
            splits.add(data_split_info)

    def test_add_non_data_split_info(self):
        splits = DataSplits(1)
        with pytest.raises(TypeError):
            splits.add("not DataSplitInfo")

    def test_get_split_0(self, data_split_info):
        splits = DataSplits(1)
        splits.add(data_split_info)
        split = splits.get_split(0)
        assert split == data_split_info

    def test_get_split_max_index(self, data_split_info):
        splits = DataSplits(2)
        splits.add(data_split_info)
        splits.add(data_split_info)
        split = splits.get_split(1)
        assert split == data_split_info

    def test_get_split_greater_than_max_index(self, data_split_info):
        splits = DataSplits(1)
        splits.add(data_split_info)
        with pytest.raises(IndexError):
            splits.get_split(1)

    def test_get_split_no_split_at_index(self):
        splits = DataSplits(1)
        with pytest.raises(ValueError):
            splits.get_split(0)
