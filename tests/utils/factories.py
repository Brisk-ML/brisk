"""Test data factories for creating Brisk objects.

This module provides factory classes for creating test objects with sensible
defaults. Factories are preferred over fixtures when you need flexibility in
parameters or need to create multiple instances with different configurations.

Usage:
    from tests.utils.factories import DataFrameFactory, AlgorithmFactory

    # Create test data with custom size
    df = DataFrameFactory.simple(rows=100, cols=5)

    # Create algorithm collection
    algorithms = AlgorithmFactory.collection(n=3)
"""
from typing import List, Optional, Tuple
from io import StringIO

import pandas as pd
import numpy as np
import sklearn.linear_model as linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn.ensemble as ensemble
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score
)

from brisk.data.data_manager import DataManager
from brisk.data.preprocessing import (
    ScalingPreprocessor,
    MissingDataPreprocessor,
    CategoricalEncodingPreprocessor,
)
from brisk.configuration import (
    AlgorithmWrapper,
    AlgorithmCollection,
    ExperimentGroup,
    Configuration
)
from brisk.evaluation import MetricWrapper, MetricManager


class DataFrameFactory:
    """Factory for creating test DataFrames with various characteristics."""

    @staticmethod
    def simple(rows: int = 5, cols: int = 2, seed: int = 42) -> pd.DataFrame:
        """Create a simple DataFrame for testing.

        Args:
            rows: Number of rows
            cols: Number of feature columns (target column added automatically)
            seed: Random seed for reproducibility

        Returns:
            DataFrame with numeric features and target column
        """
        np.random.seed(seed)
        data = {f"feature_{i}": np.random.randn(rows) for i in range(cols)}
        data["target"] = np.random.randint(0, 2, rows)
        return pd.DataFrame(data)

    @staticmethod
    def regression(
        rows: int = 10,
        noise: float = 0.1,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Create regression dataset with linear relationship.

        Args:
            rows: Number of rows
            noise: Amount of noise to add to target
            seed: Random seed

        Returns:
            Tuple of (X, y) where y = 2*x1 + 3*x2 + noise
        """
        np.random.seed(seed)
        x = pd.DataFrame({
            "x": np.random.randn(rows),
            "y": np.random.randn(rows),
        })
        # y = 2*x + 3*y + noise
        y = pd.Series(
            2 * x["x"] + 3 * x["y"] + np.random.randn(rows) * noise,
            name="target"
        )
        return x, y

    @staticmethod
    def classification(
        rows: int = 10,
        n_classes: int = 2,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Create classification dataset.

        Args:
            rows: Number of rows
            n_classes: Number of classes
            seed: Random seed

        Returns:
            Tuple of (X, y) for classification
        """
        np.random.seed(seed)
        x = pd.DataFrame({
            "feature1": np.random.randn(rows),
            "feature2": np.random.randn(rows),
        })
        y = pd.Series(np.random.randint(0, n_classes, rows), name="label")
        return x, y

    @staticmethod
    def with_missing_values(
        rows: int = 10,
        missing_pct: float = 0.2,
        seed: int = 42
    ) -> pd.DataFrame:
        """Create DataFrame with missing values.

        Args:
            rows: Number of rows
            missing_pct: Percentage of values to make missing (0.0 to 1.0)
            seed: Random seed

        Returns:
            DataFrame with missing values
        """
        np.random.seed(seed)
        df = DataFrameFactory.simple(rows, cols=3, seed=seed)
        mask = np.random.random(df.shape) < missing_pct
        df = df.mask(mask)
        return df

    @staticmethod
    def with_categorical(
        rows: int = 10,
        n_categories: int = 3,
        categorical_cols: int = 1,
        seed: int = 42
    ) -> pd.DataFrame:
        """Create DataFrame with categorical features.

        Args:
            rows: Number of rows
            n_categories: Number of unique categories
            categorical_cols: Number of categorical columns
            seed: Random seed

        Returns:
            DataFrame with both numeric and categorical features
        """
        np.random.seed(seed)
        df = DataFrameFactory.simple(rows, cols=2, seed=seed)

        for i in range(categorical_cols):
            categories = [f'Cat_{i}_{j}' for j in range(n_categories)]
            df[f'category_{i}'] = np.random.choice(categories, rows)

        return df

    @staticmethod
    def with_groups(
        rows: int = 15,
        n_groups: int = 3,
        seed: int = 42
    ) -> pd.DataFrame:
        """Create DataFrame with group column for grouped splitting.

        Args:
            rows: Number of rows
            n_groups: Number of groups
            seed: Random seed

        Returns:
            DataFrame with 'group' column
        """
        np.random.seed(seed)
        df = DataFrameFactory.simple(rows, cols=2, seed=seed)
        groups = [f"Group_{chr(65+i)}" for i in range(n_groups)]
        df["group"] = np.random.choice(groups, rows)
        return df

    @staticmethod
    def from_csv_string(csv_string: str) -> pd.DataFrame:
        """Create DataFrame from CSV string (useful for inline test data).

        Args:
            csv_string: CSV formatted string

        Returns:
            DataFrame parsed from CSV string
        """
        return pd.read_csv(StringIO(csv_string))


class AlgorithmFactory:
    """Factory for creating test algorithms and algorithm collections."""

    @staticmethod
    def linear() -> AlgorithmWrapper:
        """Create a simple linear regression algorithm."""
        return AlgorithmWrapper(
            name="linear",
            display_name="Linear Regression",
            algorithm_class=linear_model.LinearRegression
        )

    @staticmethod
    def ridge(with_hyperparams: bool = False) -> AlgorithmWrapper:
        """Create ridge regression algorithm.

        Args:
            with_hyperparams: If True, include hyperparameter grid
        """
        hyperparam_grid = {
            "alpha": [0.1, 1.0, 10.0]
        } if with_hyperparams else None
        return AlgorithmWrapper(
            name="ridge",
            display_name="Ridge Regression",
            algorithm_class=linear_model.Ridge,
            default_params={"alpha": 1.0, "max_iter": 10000},
            hyperparam_grid=hyperparam_grid
        )

    @staticmethod
    def random_forest(with_hyperparams: bool = False) -> AlgorithmWrapper:
        """Create random forest algorithm.

        Args:
            with_hyperparams: If True, include hyperparameter grid
        """
        hyperparam_grid = {
            "n_estimators": [20, 40, 60]
        } if with_hyperparams else None

        return AlgorithmWrapper(
            name="rf",
            display_name="Random Forest",
            algorithm_class=ensemble.RandomForestRegressor,
            default_params={"n_jobs": 1},
            hyperparam_grid=hyperparam_grid
        )

    @staticmethod
    def collection(
        n: int = 3,
        include_hyperparams: bool = False
    ) -> AlgorithmCollection:
        """Create an algorithm collection.

        Args:
            n: Number of algorithms (1-4)
            include_hyperparams: Whether to include hyperparameter grids

        Returns:
            AlgorithmCollection with n algorithms
        """
        algorithms = [
            AlgorithmFactory.linear(),
            AlgorithmFactory.ridge(with_hyperparams=include_hyperparams),
            AlgorithmWrapper(
                name="lasso",
                display_name="LASSO Regression",
                algorithm_class=linear_model.Lasso,
                default_params={"alpha": 0.1, "max_iter": 10000},
                hyperparam_grid={
                    "alpha": [0.1, 0.5, 1.0]
                } if include_hyperparams else None
            ),
            AlgorithmFactory.random_forest(
                with_hyperparams=include_hyperparams
            ),
        ]
        return AlgorithmCollection(*algorithms[:n])

    @staticmethod
    def classifier_collection() -> AlgorithmCollection:
        """Create a collection of classification algorithms."""
        return AlgorithmCollection(
            AlgorithmWrapper(
                name="dt",
                display_name="Decision Tree",
                algorithm_class=DecisionTreeClassifier
            ),
            AlgorithmWrapper(
                name="knn",
                display_name="K-Nearest Neighbors",
                algorithm_class=KNeighborsClassifier,
                default_params={"n_neighbors": 5}
            ),
            AlgorithmWrapper(
                name="rf_clf",
                display_name="Random Forest Classifier",
                algorithm_class=ensemble.RandomForestClassifier,
                default_params={"n_estimators": 50, "n_jobs": 1}
            )
        )


class MetricFactory:
    """Factory for creating test metrics and metric managers."""

    @staticmethod
    def mae() -> MetricWrapper:
        """Create MAE metric."""
        return MetricWrapper(
            name="mean_absolute_error",
            display_name="Mean Absolute Error",
            metric_function=mean_absolute_error,
            abbreviation="MAE"
        )

    @staticmethod
    def mse() -> MetricWrapper:
        """Create MSE metric."""
        return MetricWrapper(
            name="mean_squared_error",
            display_name="Mean Squared Error",
            metric_function=mean_squared_error,
            abbreviation="MSE"
        )

    @staticmethod
    def r2() -> MetricWrapper:
        """Create R2 metric."""
        return MetricWrapper(
            name="r2_score",
            display_name="R² Score",
            metric_function=r2_score,
            abbreviation="R²"
        )

    @staticmethod
    def regression_manager(n_metrics: int = 3) -> MetricManager:
        """Create a metric manager for regression.

        Args:
            n_metrics: Number of metrics (1-3)
        """
        metrics = [
            MetricFactory.mae(),
            MetricFactory.mse(),
            MetricFactory.r2(),
        ]
        return MetricManager(*metrics[:n_metrics])

    @staticmethod
    def classification_manager() -> MetricManager:
        """Create a metric manager for classification."""
        return MetricManager(
            MetricWrapper(
                name="accuracy_score",
                display_name="Accuracy",
                metric_function=accuracy_score,
                abbreviation="ACC"
            ),
            MetricWrapper(
                name="precision_score",
                display_name="Precision",
                metric_function=lambda y_true, y_pred: precision_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
                abbreviation="PREC"
            ),
            MetricWrapper(
                name="recall_score",
                display_name="Recall",
                metric_function=lambda y_true, y_pred: recall_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
                abbreviation="REC"
            )
        )


class DataManagerFactory:
    """Factory for creating test data managers."""

    @staticmethod
    def simple(test_size: float = 0.2, random_state: int = 42) -> DataManager:
        """Create a simple data manager.

        Args:
            test_size: Fraction of data for test set
            random_state: Random seed
        """
        return DataManager(
            test_size=test_size,
            random_state=random_state
        )

    @staticmethod
    def with_cross_validation(
        n_splits: int = 5,
        test_size: float = 0.2
    ) -> DataManager:
        """Create data manager with cross-validation.

        Args:
            n_splits: Number of CV splits
            test_size: Test set size
        """
        return DataManager(
            test_size=test_size,
            n_splits=n_splits,
            random_state=42
        )

    @staticmethod
    def with_preprocessing(
        scaling: bool = True,
        handle_missing: bool = True,
        encode_categorical: bool = False
    ) -> DataManager:
        """Create data manager with preprocessing pipeline.

        Args:
            scaling: Whether to include scaling
            handle_missing: Whether to handle missing values
            encode_categorical: Whether to encode categorical features
        """
        preprocessors = []

        if handle_missing:
            preprocessors.append(
                MissingDataPreprocessor(strategy="impute", impute_method="mean")
            )

        if scaling:
            preprocessors.append(
                ScalingPreprocessor(method="standard")
            )

        if encode_categorical:
            preprocessors.append(
                CategoricalEncodingPreprocessor(method="onehot")
            )

        return DataManager(
            test_size=0.2,
            random_state=42,
            preprocessors=preprocessors if preprocessors else None
        )


class ExperimentGroupFactory:
    """Factory for creating test experiment groups."""

    @staticmethod
    def simple(
        name: str = "test_group",
        workflow: str = "test_workflow",
        datasets: Optional[List[str]] = None,
        algorithms: Optional[List[str]] = None
    ) -> ExperimentGroup:
        """Create a simple experiment group.

        Args:
            name: Group name
            workflow: Workflow name
            datasets: List of dataset filenames
            algorithms: List of algorithm names
        """
        return ExperimentGroup(
            name=name,
            workflow=workflow,
            datasets=datasets or ["test.csv"],
            algorithms=algorithms or ["linear"]
        )

    @staticmethod
    def with_multiple_datasets(n_datasets: int = 3) -> ExperimentGroup:
        """Create experiment group with multiple datasets."""
        return ExperimentGroup(
            name="multi_dataset_group",
            workflow="test_workflow",
            datasets=[f"data{i}.csv" for i in range(n_datasets)],
            algorithms=["linear", "ridge"]
        )

    @staticmethod
    def with_preprocessing() -> ExperimentGroup:
        """Create experiment group with preprocessing configuration."""
        return ExperimentGroup(
            name="preprocessed_group",
            workflow="test_workflow",
            datasets=["test.csv"],
            algorithms=["linear"],
            data_config={
                "preprocessors": [
                    ScalingPreprocessor(method="standard"),
                    MissingDataPreprocessor(
                        strategy="impute", impute_method="mean"
                    )
                ]
            }
        )


class ConfigurationFactory:
    """Factory for creating test configurations."""

    @staticmethod
    def simple(
        workflow: str = "test_workflow",
        algorithms: Optional[List[str]] = None
    ) -> Configuration:
        """Create a simple configuration.

        Args:
            workflow: Default workflow name
            algorithms: Default algorithm names
        """
        config = Configuration(
            default_workflow=workflow,
            default_algorithms=algorithms or ["linear"]
        )
        return config

    @staticmethod
    def with_single_group(
        group_name: str = "test_group",
        datasets: Optional[List[str]] = None
    ) -> Configuration:
        """Create configuration with one experiment group."""
        config = ConfigurationFactory.simple()
        config.add_experiment_group(
            name=group_name,
            description="Test experiment group",
            datasets=datasets or ["test.csv"]
        )
        return config
