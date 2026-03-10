"""Integration tests for the preprocessing pipeline through DataManager.

Tests each of the 4 preprocessors individually and all together, using real
data loaded via DataManager.split(). These tests exercise the full path from
CSV on disk through splitting, preprocessing, and DataSplitInfo creation.
"""

import pytest
from unittest import mock

import pandas as pd
import numpy as np

from brisk.configuration import project
from brisk.data import data_manager
from brisk.data.data_split_info import DataSplitInfo
from brisk.data.preprocessing import (
    MissingDataPreprocessor,
    ScalingPreprocessor,
    CategoricalEncodingPreprocessor,
    FeatureSelectionPreprocessor,
)
from brisk import services
from brisk.services import missing

# pylint: disable=W0621, W0212


@pytest.fixture()
def mock_services(tmp_path):
    """Service bundle with mocked reporting for integration tests."""
    return services.bundle.ServiceBundle(
        io=services.io.IOService("io", tmp_path, tmp_path),
        logger=missing.MissingServices(),
        metadata=missing.MissingServices(),
        utility=missing.MissingServices(),
        reporting=mock.Mock(),
        rerun=missing.MissingServices()
    )


@pytest.fixture()
def numeric_csv(tmp_path):
    """CSV with only numeric features, no missing values."""
    np.random.seed(42)
    df = pd.DataFrame({
        "feat_a": np.random.randn(50) * 10 + 50,
        "feat_b": np.random.randn(50) * 5 + 20,
        "feat_c": np.random.randn(50) * 2 + 100,
        "target": np.random.choice([0, 1], size=50),
    })
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir(parents=True)
    path = datasets_dir / "numeric.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def missing_csv(tmp_path):
    """CSV with NaN values scattered across numeric features."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "feat_a": np.random.randn(n) * 10 + 50,
        "feat_b": np.random.randn(n) * 5 + 20,
        "feat_c": np.random.randn(n) * 2 + 100,
        "target": np.random.choice([0, 1], size=n),
    })
    for idx in [0, 3, 5, 10, 15, 20, 25, 30]:
        col = ["feat_a", "feat_b", "feat_c"][idx % 3]
        df.loc[idx, col] = np.nan

    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir(parents=True)
    path = datasets_dir / "missing.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def categorical_csv(tmp_path):
    """CSV with one categorical and one numeric feature."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "num_feat": np.random.randn(n) * 10 + 50,
        "cat_feat": np.random.choice(["A", "B", "C"], size=n),
        "target": np.random.choice([0, 1], size=n),
    })
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir(parents=True)
    path = datasets_dir / "categorical.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def multi_feature_csv(tmp_path):
    """CSV with enough numeric features for feature selection."""
    np.random.seed(42)
    n = 80
    df = pd.DataFrame({
        "feat_0": np.random.randn(n),
        "feat_1": np.random.randn(n),
        "feat_2": np.random.randn(n),
        "feat_3": np.random.randn(n),
        "feat_4": np.random.randn(n),
        "feat_5": np.random.randn(n),
        "target": np.random.choice([0, 1], size=n),
    })
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir(parents=True)
    path = datasets_dir / "multi_feature.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def mixed_csv(tmp_path):
    """CSV with numeric features, categorical feature, and missing values."""
    np.random.seed(42)
    n = 80
    df = pd.DataFrame({
        "num_1": np.random.randn(n) * 10 + 50,
        "num_2": np.random.randn(n) * 5 + 20,
        "num_3": np.random.randn(n) * 2 + 100,
        "num_4": np.random.randn(n) * 3 + 30,
        "num_5": np.random.randn(n) * 7 + 10,
        "cat_1": np.random.choice(["X", "Y", "Z"], size=n),
        "target": np.random.choice([0, 1], size=n),
    })
    for idx in [0, 5, 10, 15, 20, 25]:
        col = ["num_1", "num_2", "num_3"][idx % 3]
        df.loc[idx, col] = np.nan

    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir(parents=True)
    path = datasets_dir / "mixed.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# MissingDataPreprocessor
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestMissingDataPreprocessorIntegration:
    """Test MissingDataPreprocessor through DataManager.split()."""

    @pytest.fixture(autouse=True, scope="class")
    def patch_slow_methods(self):
        with (
            mock.patch.object(
                DataSplitInfo, "set_services", return_value=None
            ),
            mock.patch.object(
                DataSplitInfo, "evaluate_data_split", return_value=None
            ),
        ):
            yield

    def test_impute_mean_removes_all_nan(
        self, tmp_path, missing_csv, mock_services
    ):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[
                    MissingDataPreprocessor(
                        strategy="impute", impute_method="mean"
                    )
                ],
            )
            manager.set_services(mock_services)
            splits = manager.split(missing_csv, [], "g", "missing")

        split = splits.get_split(0)
        assert not split.X_train.isnull().any().any()
        assert not split.X_test.isnull().any().any()

    def test_impute_median_removes_all_nan(
        self, tmp_path, missing_csv, mock_services
    ):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[
                    MissingDataPreprocessor(
                        strategy="impute", impute_method="median"
                    )
                ],
            )
            manager.set_services(mock_services)
            splits = manager.split(missing_csv, [], "g", "missing_med")

        split = splits.get_split(0)
        assert not split.X_train.isnull().any().any()
        assert not split.X_test.isnull().any().any()

    def test_drop_rows_aligns_x_and_y(
        self, tmp_path, missing_csv, mock_services
    ):
        """After drop_rows, X and y must have the same number of rows."""
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[
                    MissingDataPreprocessor(strategy="drop_rows")
                ],
            )
            manager.set_services(mock_services)
            splits = manager.split(missing_csv, [], "g", "missing_drop")

        split = splits.get_split(0)
        assert not split.X_train.isnull().any().any()
        assert not split.X_test.isnull().any().any()
        assert len(split.X_train) == len(split.y_train), (
            f"X_train ({len(split.X_train)}) and y_train "
            f"({len(split.y_train)}) length mismatch after drop_rows"
        )
        assert len(split.X_test) == len(split.y_test), (
            f"X_test ({len(split.X_test)}) and y_test "
            f"({len(split.y_test)}) length mismatch after drop_rows"
        )

    def test_impute_preserves_row_count(
        self, tmp_path, missing_csv, mock_services
    ):
        """Imputation should not change the number of rows."""
        with project.ProjectRootContext(tmp_path):
            manager_raw = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[
                    MissingDataPreprocessor(
                        strategy="impute", impute_method="mean"
                    )
                ],
            )
            manager_raw.set_services(mock_services)
            splits = manager_raw.split(missing_csv, [], "g", "missing_cnt")

        split = splits.get_split(0)
        assert len(split.X_train) == len(split.y_train)
        assert len(split.X_test) == len(split.y_test)

    def test_no_missing_preprocessor_with_other_preprocessors_raises(
        self, tmp_path, missing_csv, mock_services
    ):
        """Having other preprocessors but no MissingDataPreprocessor should
        raise when missing values are present."""
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[ScalingPreprocessor(method="standard")],
            )
            manager.set_services(mock_services)
            with pytest.raises(ValueError, match="Missing values detected"):
                manager.split(missing_csv, [], "g", "missing_err")

    def test_impute_multiple_splits(
        self, tmp_path, missing_csv, mock_services
    ):
        """Imputation should work correctly across all splits."""
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=3, test_size=0.3, random_state=42,
                preprocessors=[
                    MissingDataPreprocessor(
                        strategy="impute", impute_method="mean"
                    )
                ],
            )
            manager.set_services(mock_services)
            splits = manager.split(missing_csv, [], "g", "missing_multi")

        for i in range(3):
            split = splits.get_split(i)
            assert not split.X_train.isnull().any().any(), (
                f"Split {i}: X_train still has NaN"
            )
            assert not split.X_test.isnull().any().any(), (
                f"Split {i}: X_test still has NaN"
            )
            assert len(split.X_train) == len(split.y_train), (
                f"Split {i}: X/y train length mismatch"
            )
            assert len(split.X_test) == len(split.y_test), (
                f"Split {i}: X/y test length mismatch"
            )


# ---------------------------------------------------------------------------
# ScalingPreprocessor
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestScalingPreprocessorIntegration:
    """Test ScalingPreprocessor through DataManager.split()."""

    @pytest.fixture(autouse=True, scope="class")
    def patch_slow_methods(self):
        with (
            mock.patch.object(
                DataSplitInfo, "set_services", return_value=None
            ),
            mock.patch.object(
                DataSplitInfo, "evaluate_data_split", return_value=None
            ),
        ):
            yield

    def test_standard_scaling_transforms_values(
        self, tmp_path, numeric_csv, mock_services
    ):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[ScalingPreprocessor(method="standard")],
            )
            manager.set_services(mock_services)
            splits = manager.split(numeric_csv, [], "g", "numeric_std")

        split = splits.get_split(0)
        for col in split.X_train.columns:
            assert abs(split.X_train[col].mean()) < 0.5, (
                f"Training mean of {col} should be near 0"
            )
            assert 0.5 < split.X_train[col].std() < 1.5, (
                f"Training std of {col} should be near 1"
            )

    def test_minmax_scaling(self, tmp_path, numeric_csv, mock_services):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[ScalingPreprocessor(method="minmax")],
            )
            manager.set_services(mock_services)
            splits = manager.split(numeric_csv, [], "g", "numeric_mm")

        split = splits.get_split(0)
        for col in split.X_train.columns:
            assert split.X_train[col].min() >= -0.01
            assert split.X_train[col].max() <= 1.01

    def test_scaling_preserves_shape(
        self, tmp_path, numeric_csv, mock_services
    ):
        with project.ProjectRootContext(tmp_path):
            manager_no = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
            )
            manager_no.set_services(mock_services)
            splits_no = manager_no.split(numeric_csv, [], "g", "no_scale")

            manager_yes = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[ScalingPreprocessor(method="standard")],
            )
            manager_yes.set_services(mock_services)
            splits_yes = manager_yes.split(numeric_csv, [], "g", "yes_scale")

        s_no = splits_no.get_split(0)
        s_yes = splits_yes.get_split(0)
        assert s_no.X_train.shape == s_yes.X_train.shape
        assert s_no.X_test.shape == s_yes.X_test.shape

    def test_scaler_stored_in_split(
        self, tmp_path, numeric_csv, mock_services
    ):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[ScalingPreprocessor(method="standard")],
            )
            manager.set_services(mock_services)
            splits = manager.split(numeric_csv, [], "g", "scaler_store")

        assert splits.get_split(0).scaler is not None

    def test_scaling_excludes_categorical_features(
        self, tmp_path, categorical_csv, mock_services
    ):
        """Categorical features should NOT be scaled even after encoding."""
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[
                    CategoricalEncodingPreprocessor(method="label"),
                    ScalingPreprocessor(method="standard"),
                ],
            )
            manager.set_services(mock_services)
            splits = manager.split(
                categorical_csv, ["cat_feat"], "g", "scale_cat"
            )

        split = splits.get_split(0)
        cat_values = set(split.X_train["cat_feat"].unique())
        assert cat_values.issubset({0, 1, 2}), (
            f"Label-encoded cat_feat should be {{0,1,2}}, got {cat_values}. "
            "It may have been incorrectly scaled."
        )

    def test_scaling_multiple_splits(
        self, tmp_path, numeric_csv, mock_services
    ):
        """Scaling should work correctly across all splits."""
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=3, test_size=0.3, random_state=42,
                preprocessors=[ScalingPreprocessor(method="standard")],
            )
            manager.set_services(mock_services)
            splits = manager.split(numeric_csv, [], "g", "scale_multi")

        for i in range(3):
            split = splits.get_split(i)
            for col in split.X_train.columns:
                assert abs(split.X_train[col].mean()) < 0.5, (
                    f"Split {i}, col {col}: mean not near 0"
                )
            assert len(split.X_train) == len(split.y_train), (
                f"Split {i}: X/y train length mismatch"
            )


# ---------------------------------------------------------------------------
# CategoricalEncodingPreprocessor
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestCategoricalEncodingIntegration:
    """Test CategoricalEncodingPreprocessor through DataManager.split()."""

    @pytest.fixture(autouse=True, scope="class")
    def patch_slow_methods(self):
        with (
            mock.patch.object(
                DataSplitInfo, "set_services", return_value=None
            ),
            mock.patch.object(
                DataSplitInfo, "evaluate_data_split", return_value=None
            ),
        ):
            yield

    def test_onehot_creates_binary_columns(
        self, tmp_path, categorical_csv, mock_services
    ):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[
                    CategoricalEncodingPreprocessor(method="onehot")
                ],
            )
            manager.set_services(mock_services)
            splits = manager.split(
                categorical_csv, ["cat_feat"], "g", "onehot"
            )

        split = splits.get_split(0)
        assert "cat_feat" not in split.X_train.columns
        onehot_cols = [
            c for c in split.X_train.columns if c.startswith("cat_feat_")
        ]
        assert len(onehot_cols) >= 2
        for col in onehot_cols:
            assert set(split.X_train[col].unique()).issubset({0.0, 1.0})

    def test_label_encoding_produces_integers(
        self, tmp_path, categorical_csv, mock_services
    ):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[
                    CategoricalEncodingPreprocessor(method="label")
                ],
            )
            manager.set_services(mock_services)
            splits = manager.split(
                categorical_csv, ["cat_feat"], "g", "label_enc"
            )

        split = splits.get_split(0)
        assert "cat_feat" in split.X_train.columns
        values = set(split.X_train["cat_feat"].unique())
        assert values.issubset({0, 1, 2})

    def test_ordinal_encoding(self, tmp_path, categorical_csv, mock_services):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[
                    CategoricalEncodingPreprocessor(method="ordinal")
                ],
            )
            manager.set_services(mock_services)
            splits = manager.split(
                categorical_csv, ["cat_feat"], "g", "ordinal_enc"
            )

        split = splits.get_split(0)
        assert "cat_feat" in split.X_train.columns
        assert split.X_train["cat_feat"].dtype in [
            np.float64, np.int64, np.int32
        ]

    def test_encoding_preserves_numeric_feature(
        self, tmp_path, categorical_csv, mock_services
    ):
        with project.ProjectRootContext(tmp_path):
            manager_raw = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
            )
            manager_raw.set_services(mock_services)
            splits_raw = manager_raw.split(
                categorical_csv, ["cat_feat"], "g", "raw"
            )

            manager_enc = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[
                    CategoricalEncodingPreprocessor(method="label")
                ],
            )
            manager_enc.set_services(mock_services)
            splits_enc = manager_enc.split(
                categorical_csv, ["cat_feat"], "g", "enc"
            )

        raw = splits_raw.get_split(0)
        enc = splits_enc.get_split(0)
        pd.testing.assert_series_equal(
            raw.X_train["num_feat"].reset_index(drop=True),
            enc.X_train["num_feat"].reset_index(drop=True),
            check_names=False,
        )

    def test_onehot_multiple_splits_all_encoded(
        self, tmp_path, categorical_csv, mock_services
    ):
        """Every split should have one-hot columns, not the raw category."""
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=3, test_size=0.3, random_state=42,
                preprocessors=[
                    CategoricalEncodingPreprocessor(method="onehot")
                ],
            )
            manager.set_services(mock_services)
            splits = manager.split(
                categorical_csv, ["cat_feat"], "g", "onehot_multi"
            )

        for i in range(3):
            split = splits.get_split(i)
            assert "cat_feat" not in split.X_train.columns, (
                f"Split {i}: raw cat_feat should not exist after one-hot"
            )
            onehot_cols = [
                c for c in split.X_train.columns if c.startswith("cat_feat_")
            ]
            assert len(onehot_cols) >= 2, (
                f"Split {i}: expected one-hot columns, got {list(split.X_train.columns)}"
            )
            assert list(split.X_train.columns) == list(split.X_test.columns), (
                f"Split {i}: train/test columns differ"
            )

    def test_label_encoding_multiple_splits(
        self, tmp_path, categorical_csv, mock_services
    ):
        """Label encoding should work on every split, not just the first."""
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=3, test_size=0.3, random_state=42,
                preprocessors=[
                    CategoricalEncodingPreprocessor(method="label")
                ],
            )
            manager.set_services(mock_services)
            splits = manager.split(
                categorical_csv, ["cat_feat"], "g", "label_multi"
            )

        for i in range(3):
            split = splits.get_split(i)
            assert "cat_feat" in split.X_train.columns, (
                f"Split {i}: cat_feat should still be present"
            )
            values = set(split.X_train["cat_feat"].unique())
            assert values.issubset({0, 1, 2}), (
                f"Split {i}: label-encoded values should be integers, "
                f"got {values}"
            )


# ---------------------------------------------------------------------------
# FeatureSelectionPreprocessor
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestFeatureSelectionIntegration:
    """Test FeatureSelectionPreprocessor through DataManager.split()."""

    @pytest.fixture(autouse=True, scope="class")
    def patch_slow_methods(self):
        with (
            mock.patch.object(
                DataSplitInfo, "set_services", return_value=None
            ),
            mock.patch.object(
                DataSplitInfo, "evaluate_data_split", return_value=None
            ),
        ):
            yield

    def test_selectkbest_reduces_feature_count(
        self, tmp_path, multi_feature_csv, mock_services
    ):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[
                    FeatureSelectionPreprocessor(
                        method="selectkbest",
                        n_features_to_select=3,
                        problem_type="classification",
                    )
                ],
            )
            manager.set_services(mock_services)
            splits = manager.split(multi_feature_csv, [], "g", "fs")

        split = splits.get_split(0)
        assert split.X_train.shape[1] == 3
        assert split.X_test.shape[1] == 3

    def test_train_test_same_columns_after_selection(
        self, tmp_path, multi_feature_csv, mock_services
    ):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[
                    FeatureSelectionPreprocessor(
                        method="selectkbest",
                        n_features_to_select=3,
                        problem_type="classification",
                    )
                ],
            )
            manager.set_services(mock_services)
            splits = manager.split(multi_feature_csv, [], "g", "fs_cols")

        split = splits.get_split(0)
        assert list(split.X_train.columns) == list(split.X_test.columns)

    def test_selected_features_are_subset_of_original(
        self, tmp_path, multi_feature_csv, mock_services
    ):
        original_features = {f"feat_{i}" for i in range(6)}
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[
                    FeatureSelectionPreprocessor(
                        method="selectkbest",
                        n_features_to_select=3,
                        problem_type="classification",
                    )
                ],
            )
            manager.set_services(mock_services)
            splits = manager.split(multi_feature_csv, [], "g", "fs_sub")

        selected = set(splits.get_split(0).X_train.columns)
        assert selected.issubset(original_features)

    def test_feature_selection_multiple_splits(
        self, tmp_path, multi_feature_csv, mock_services
    ):
        """Feature selection should produce valid results for every split."""
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=3, test_size=0.3, random_state=42,
                preprocessors=[
                    FeatureSelectionPreprocessor(
                        method="selectkbest",
                        n_features_to_select=3,
                        problem_type="classification",
                    )
                ],
            )
            manager.set_services(mock_services)
            splits = manager.split(multi_feature_csv, [], "g", "fs_multi")

        for i in range(3):
            split = splits.get_split(i)
            assert split.X_train.shape[1] == 3, (
                f"Split {i}: expected 3 features, got {split.X_train.shape[1]}"
            )
            assert split.X_test.shape[1] == 3, (
                f"Split {i}: test set should also have 3 features"
            )
            assert list(split.X_train.columns) == list(split.X_test.columns), (
                f"Split {i}: train/test column mismatch"
            )


# ---------------------------------------------------------------------------
# All 4 Preprocessors Combined
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestAllPreprocessorsIntegration:
    """Test all 4 preprocessors working together through DataManager."""

    @pytest.fixture(autouse=True, scope="class")
    def patch_slow_methods(self):
        with (
            mock.patch.object(
                DataSplitInfo, "set_services", return_value=None
            ),
            mock.patch.object(
                DataSplitInfo, "evaluate_data_split", return_value=None
            ),
        ):
            yield

    def test_full_pipeline_single_split(
        self, tmp_path, mixed_csv, mock_services
    ):
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[
                    MissingDataPreprocessor(
                        strategy="impute", impute_method="mean"
                    ),
                    CategoricalEncodingPreprocessor(method="onehot"),
                    ScalingPreprocessor(method="standard"),
                    FeatureSelectionPreprocessor(
                        method="selectkbest",
                        n_features_to_select=3,
                        problem_type="classification",
                    ),
                ],
            )
            manager.set_services(mock_services)
            splits = manager.split(
                mixed_csv, ["cat_1"], "g", "full_pipe"
            )

        split = splits.get_split(0)
        assert not split.X_train.isnull().any().any()
        assert not split.X_test.isnull().any().any()
        assert split.X_train.shape[1] == 3
        assert split.X_test.shape[1] == 3
        assert list(split.X_train.columns) == list(split.X_test.columns)
        assert len(split.X_train) == len(split.y_train)
        assert len(split.X_test) == len(split.y_test)

    def test_full_pipeline_multiple_splits(
        self, tmp_path, mixed_csv, mock_services
    ):
        """Every split must be preprocessed identically in structure."""
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=3, test_size=0.3, random_state=42,
                preprocessors=[
                    MissingDataPreprocessor(
                        strategy="impute", impute_method="mean"
                    ),
                    CategoricalEncodingPreprocessor(method="onehot"),
                    ScalingPreprocessor(method="standard"),
                    FeatureSelectionPreprocessor(
                        method="selectkbest",
                        n_features_to_select=3,
                        problem_type="classification",
                    ),
                ],
            )
            manager.set_services(mock_services)
            splits = manager.split(
                mixed_csv, ["cat_1"], "g", "full_multi"
            )

        for i in range(3):
            split = splits.get_split(i)
            assert not split.X_train.isnull().any().any(), (
                f"Split {i}: NaN in X_train"
            )
            assert not split.X_test.isnull().any().any(), (
                f"Split {i}: NaN in X_test"
            )
            assert split.X_train.shape[1] == 3, (
                f"Split {i}: expected 3 features, "
                f"got {split.X_train.shape[1]}"
            )
            assert split.X_test.shape[1] == 3, (
                f"Split {i}: expected 3 test features"
            )
            assert list(split.X_train.columns) == list(split.X_test.columns), (
                f"Split {i}: train/test column mismatch"
            )
            assert len(split.X_train) == len(split.y_train), (
                f"Split {i}: X/y train length mismatch"
            )
            assert len(split.X_test) == len(split.y_test), (
                f"Split {i}: X/y test length mismatch"
            )

    def test_pipeline_order_independent_of_list_order(
        self, tmp_path, mixed_csv, mock_services
    ):
        """DataManager applies preprocessors in a fixed order regardless of
        the order they appear in the list."""
        with project.ProjectRootContext(tmp_path):
            manager_forward = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[
                    MissingDataPreprocessor(
                        strategy="impute", impute_method="mean"
                    ),
                    CategoricalEncodingPreprocessor(method="onehot"),
                    ScalingPreprocessor(method="standard"),
                ],
            )
            manager_forward.set_services(mock_services)
            splits_fwd = manager_forward.split(
                mixed_csv, ["cat_1"], "g", "order_fwd"
            )

            manager_reverse = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[
                    ScalingPreprocessor(method="standard"),
                    CategoricalEncodingPreprocessor(method="onehot"),
                    MissingDataPreprocessor(
                        strategy="impute", impute_method="mean"
                    ),
                ],
            )
            manager_reverse.set_services(mock_services)
            splits_rev = manager_reverse.split(
                mixed_csv, ["cat_1"], "g", "order_rev"
            )

        fwd = splits_fwd.get_split(0)
        rev = splits_rev.get_split(0)
        assert set(fwd.X_train.columns) == set(rev.X_train.columns)
        pd.testing.assert_frame_equal(
            fwd.X_train.sort_index(axis=1).reset_index(drop=True),
            rev.X_train.sort_index(axis=1).reset_index(drop=True),
        )

    def test_label_encoding_then_scaling_preserves_categories(
        self, tmp_path, categorical_csv, mock_services
    ):
        """When label encoding + scaling are both applied, the label-encoded
        feature should NOT be scaled (it is still categorical)."""
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[
                    CategoricalEncodingPreprocessor(method="label"),
                    ScalingPreprocessor(method="standard"),
                ],
            )
            manager.set_services(mock_services)
            splits = manager.split(
                categorical_csv, ["cat_feat"], "g", "label_scale"
            )

        split = splits.get_split(0)
        cat_values = set(split.X_train["cat_feat"].unique())
        assert cat_values.issubset({0, 1, 2}), (
            f"Label-encoded cat_feat values should be {{0,1,2}}, "
            f"got {cat_values} — the feature was incorrectly scaled."
        )

    def test_onehot_encoding_then_scaling_not_scaled(
        self, tmp_path, categorical_csv, mock_services
    ):
        """One-hot encoded columns should remain binary (0/1) and not be
        scaled."""
        with project.ProjectRootContext(tmp_path):
            manager = data_manager.DataManager(
                n_splits=1, test_size=0.3, random_state=42,
                preprocessors=[
                    CategoricalEncodingPreprocessor(method="onehot"),
                    ScalingPreprocessor(method="standard"),
                ],
            )
            manager.set_services(mock_services)
            splits = manager.split(
                categorical_csv, ["cat_feat"], "g", "onehot_scale"
            )

        split = splits.get_split(0)
        onehot_cols = [
            c for c in split.X_train.columns if c.startswith("cat_feat_")
        ]
        for col in onehot_cols:
            vals = set(split.X_train[col].unique())
            assert vals.issubset({0.0, 1.0}), (
                f"One-hot column {col} should be binary, got {vals}. "
                "It was incorrectly scaled."
            )
