"""Integration tests for DataManager."""

import pytest
from unittest import mock
import pandas as pd
import numpy as np
import sqlite3

from brisk.configuration import project
from brisk.data import data_manager, splitkey
from brisk import services
from brisk.services import missing
from brisk.data.data_split_info import DataSplitInfo

# pylint: disable=W0621, W0212, W0613, W0612

@pytest.fixture()
def sample_df():
    return pd.DataFrame(
        np.random.randint(0, 100, size=(10, 3)),
        columns=["col_a", "col_b", "col_c"]
    )


@pytest.fixture()
def sample_df_with_groups():
    group_ids = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7]
    return pd.DataFrame({
        "col_a": np.random.randint(0, 100, size=20),
        "col_b": np.random.randint(0, 100, size=20),
        "group_id": group_ids,
        "col_c": np.random.randint(0, 100, size=20)
    })


@pytest.fixture()
def csv_file(tmp_path, sample_df):
    with project.ProjectRootContext(tmp_path):
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True)

        file_path = datasets_dir / "test.csv"
        sample_df.to_csv(file_path, index=False)


@pytest.fixture()
def csv_file_with_groups(tmp_path, sample_df_with_groups):
    with project.ProjectRootContext(tmp_path):
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True)

        file_path = datasets_dir / "test_groups.csv"
        sample_df_with_groups.to_csv(file_path, index=False)


@pytest.fixture()
def sqlite_db(tmp_path, sample_df):
    with project.ProjectRootContext(tmp_path):
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True)

        db_path = datasets_dir / "test.db"

        with sqlite3.connect(db_path) as conn:
            sample_df.to_sql("test_table", conn, index=False)


@pytest.fixture()
def mock_services(tmp_path):
    """Create a service bundle with mocked rerun service."""
    return services.bundle.ServiceBundle(
        io=services.io.IOService("io", tmp_path, tmp_path),
        logger=missing.MissingServices(),
        metadata=missing.MissingServices(),
        utility=missing.MissingServices(),
        reporting=mock.Mock(),
        rerun=missing.MissingServices()
    )


@pytest.mark.integration
class TestDataManagerIntegration:
    """Integration tests for the DataManager class."""

    @pytest.fixture(autouse=True, scope="class")
    def patch_slow_methods(self):
        """Patch time-consuming DataSplitInfo methods for tests."""
        with (
            mock.patch.object(
                DataSplitInfo, "set_services", return_value=None
            ),
            mock.patch.object(
                DataSplitInfo, "evaluate_data_split", return_value=None
            )
        ):
            yield

    def test_split_no_table_name(self, tmp_path, csv_file, mock_services):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager()
            manager.set_services(mock_services)
            manager.split(
                tmp_path / "datasets" / "test.csv",
                [],
                "group1",
                "test"
            )

        created_key = list(manager._splits.keys())[0]
        assert created_key == splitkey.SplitKey("group1", "test", None)

    def test_split_table_name(self, tmp_path, mock_services, sqlite_db):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager()
            manager.set_services(mock_services)
            splits = manager.split(
                tmp_path / "datasets" / "test.db",
                [],
                "group1",
                "test",
                table_name="test_table"
            )

        created_key = list(manager._splits.keys())[0]
        assert created_key == splitkey.SplitKey("group1", "test", "test_table")
        assert len(splits) == 5

    def test_split_0_splits(self, tmp_path, mock_services, csv_file):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(n_splits=0)
            manager.set_services(mock_services)

            with pytest.raises(ValueError):
                splits = manager.split(
                    tmp_path / "datasets" / "test.csv",
                    [],
                    "group1",
                    "test"
                )

    def test_split_1_split(self, tmp_path, mock_services, csv_file):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(n_splits=1, test_size=0.3)
            manager.set_services(mock_services)
            splits = manager.split(
                tmp_path / "datasets" / "test.csv",
                [],
                "group1",
                "test"
            )

        assert len(splits) == 1

    def test_split_2_splits(self, tmp_path, mock_services, csv_file):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(n_splits=2, test_size=0.3)
            manager.set_services(mock_services)
            splits = manager.split(
                tmp_path / "datasets" / "test.csv",
                [],
                "group1",
                "test"
            )

        assert len(splits) == 2

    def test_split_drop_group_column(
        self,
        tmp_path,
        mock_services,
        csv_file_with_groups
    ):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=1,
                group_column="group_id"
            )
            manager.set_services(mock_services)
            splits = manager.split(
                tmp_path / "datasets" / "test_groups.csv",
                [],
                "group1",
                "test"
            )

        split = splits.get_split(0)

        # Verify group_id is not in training/test features
        assert "group_id" not in split.X_train.columns
        assert "group_id" not in split.X_test.columns

        # Verify group indices were properly stored
        assert split.group_index_train is not None
        assert split.group_index_test is not None
        assert "values" in split.group_index_train
        assert "indices" in split.group_index_train
        assert "series" in split.group_index_train

    def test_split_caching(self, tmp_path, csv_file, mock_services):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager()
            manager.set_services(mock_services)

            splits1 = manager.split(
                tmp_path / "datasets" / "test.csv",
                [],
                "group1",
                "test"
            )

            splits2 = manager.split(
                tmp_path / "datasets" / "test.csv",
                [],
                "group1",
                "test"
            )

        assert splits1 is splits2

    def test_split_different_keys_not_cached(
        self, tmp_path, csv_file, mock_services
    ):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager()
            manager.set_services(mock_services)

            splits1 = manager.split(
                tmp_path / "datasets" / "test.csv",
                [],
                "group1",
                "test1"
            )

            splits2 = manager.split(
                tmp_path / "datasets" / "test.csv",
                [],
                "group1",
                "test2"
            )

        assert splits1 is not splits2
        assert len(manager._splits) == 2

    def test_split_categorical_features_handled(self, tmp_path, mock_services):
        with project.ProjectRootContext(tmp_path):
            cat_vals = ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]
            df = pd.DataFrame({
                "cat_feature": cat_vals,
                "num_feature": range(10),
                "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            })

            datasets_dir = tmp_path / "datasets"
            datasets_dir.mkdir(parents=True)
            file_path = datasets_dir / "test_cat.csv"
            df.to_csv(file_path, index=False)

            manager = data_manager.DataManager()
            manager.set_services(mock_services)
            splits = manager.split(
                file_path,
                categorical_features=["cat_feature"],
                group_name="group1",
                filename="test_cat"
            )

        split = splits.get_split(0)
        assert "cat_feature" in split.categorical_features
