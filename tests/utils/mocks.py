"""Mock objects for solitary unit testing.

This module provides mock implementations of Brisk classes and services to
enable fast, isolated unit tests without requiring the full system setup.

Usage:
    from tests.utils.mocks import MockServiceBundle, MockModel

    services = MockServiceBundle()
    model = MockModel(predictions=[0, 1, 0, 1])
    model.fit(X_train, y_train)
"""
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================================
# Service Mocks
# ============================================================================

class MockBaseService:
    """Base mock service class."""

    def __init__(self, name: str):
        """Initialize mock service.

        Args:
            name: Service name
        """
        self.name = name
        self._other_services = {}

    def register_services(self, services: Dict[str, Any]):
        """Register other services."""
        self._other_services = services

    def get_service(self, name: str):
        """Get another service by name."""
        return self._other_services.get(name)


class MockLoggingService(MockBaseService):
    """Mock logging service for testing.

    Captures all log messages for assertions without printing to console.
    Mimics the interface of brisk.services.logging.LoggingService.

    Example:
        logger = MockLoggingService()
        logger.logger.info("test message")
        assert ("INFO", "test message") in logger.messages
    """

    def __init__(self, name: str = "logger"):
        super().__init__(name)
        self.messages: List[Tuple[str, str]] = []
        self.results_dir = None
        self.verbose = False
        self.logger = self._create_mock_logger()

    def _create_mock_logger(self):
        """Create a mock logger with standard logging methods."""
        mock_logger = type("MockLogger", (), {})()
        mock_logger.info = lambda msg: self.messages.append(("INFO", msg))
        mock_logger.warning = lambda msg: self.messages.append(("WARNING", msg))
        mock_logger.error = lambda msg: self.messages.append(("ERROR", msg))
        mock_logger.debug = lambda msg: self.messages.append(("DEBUG", msg))
        return mock_logger

    def setup_logger(self):
        """Mock setup logger (no-op)."""
        pass

    def set_results_dir(self, results_dir: Path):
        """Mock set results directory."""
        self.results_dir = results_dir
        self.setup_logger()

    def clear(self):
        """Clear all logged messages."""
        self.messages.clear()


class MockIOService(MockBaseService):
    """Mock I/O service for testing.

    Tracks all save operations without actually writing to disk.
    Mimics the interface of brisk.services.io.IOService.

    Example:
        io_service = MockIOService()
        io_service.save_plot(Path("output.png"), plot=my_plot)
        assert len(io_service.saved_plots) == 1
    """

    def __init__(self, name: str = "io"):
        super().__init__(name)
        self.results_dir = None
        self.output_dir = None
        self.saved_plots: List[Tuple[Path, Any]] = []
        self.saved_json: List[Tuple[Path, Dict]] = []
        self.saved_rerun_configs: List[Tuple[Path, Dict]] = []
        self.io_settings = {
            "format": "png",
            "width": 10,
            "height": 8,
            "dpi": 300,
            "transparent": False
        }

    def set_output_dir(self, output_dir: Path):
        """Set the output directory (no-op in mock)."""
        self.output_dir = output_dir

    def save_to_json(self, data: Dict, output_path: Path, metadata: Dict):
        """Mock saving JSON data."""
        self.saved_json.append((output_path, data))

    def save_plot(self, output_path: Path, **kwargs):
        """Mock saving a plot."""
        plot = kwargs.get("plot")
        self.saved_plots.append((output_path, plot))

    def save_rerun_config(self, output_path: Path, config: Dict):
        """Mock saving rerun configuration."""
        self.saved_rerun_configs.append((output_path, config))

    def set_io_settings(self, io_settings: Dict[str, Any]):
        """Set I/O settings."""
        self.io_settings.update(io_settings)

    def load_data(self, data_path: Path, **kwargs):
        """Mock loading data (returns None)."""
        return None

    def load_module_object(self, file_path: Path, variable_name: str, **kwargs):
        """Mock loading module object (returns None)."""
        return None

    def load_custom_evaluators(self, evaluators_file: Path):
        """Mock loading custom evaluators (returns empty list)."""
        return []

    def load_base_data_manager(self, data_file: Path):
        """Mock loading base data manager (returns None)."""
        return None

    def load_algorithms(self, algorithm_file: Path):
        """Mock loading algorithms (returns None)."""
        return None

    def load_workflow(self, workflow_name: str):
        """Mock loading workflow (returns None)."""
        return None

    def load_metric_config(self, metric_file: Path):
        """Mock loading metric config (returns None)."""
        return None

    def clear(self):
        """Clear all saved items."""
        self.saved_plots.clear()
        self.saved_json.clear()
        self.saved_rerun_configs.clear()


class MockMetadataService(MockBaseService):
    """Mock metadata service for testing.

    Mimics the interface of brisk.services.metadata.MetadataService.
    """

    def __init__(self, name: str = "metadata"):
        super().__init__(name)
        self.algorithm_config = None
        self.generated_metadata: List[Dict[str, Any]] = []

    def set_algorithm_config(self, algorithm_config: Any):
        """Set algorithm configuration."""
        self.algorithm_config = algorithm_config

    def get_model(
        self,
        models: Any,
        method_name: str,
        is_test: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """Mock get model metadata."""
        metadata = {
            "method_name": method_name,
            "is_test": is_test,
            "timestamp": "2024-01-01T00:00:00"
        }
        self.generated_metadata.append(metadata)
        return metadata

    def get_dataset(
        self,
        method_name: str,
        dataset_name: str,
        group_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Mock get dataset metadata."""
        metadata = {
            "method_name": method_name,
            "dataset_name": dataset_name,
            "group_name": group_name,
            "timestamp": "2024-01-01T00:00:00"
        }
        self.generated_metadata.append(metadata)
        return metadata

    def get_rerun(self, method_name: str) -> Dict[str, Any]:
        """Mock get rerun metadata."""
        metadata = {
            "method_name": method_name,
            "timestamp": "2024-01-01T00:00:00"
        }
        self.generated_metadata.append(metadata)
        return metadata

    def clear(self):
        """Clear all metadata."""
        self.generated_metadata.clear()


class MockReportingService(MockBaseService):
    """Mock reporting service for testing.

    Tracks all report data additions without generating actual reports.
    Mimics the interface of brisk.services.reporting.ReportingService.

    Example:
        reporting = MockReportingService()
        reporting.set_context("group1", "dataset1", 0)
        assert reporting.get_context() == ("group1", "dataset1", 0)
    """

    def __init__(self, name: str = "reporting"):
        super().__init__(name)
        self.contexts: List[Tuple[str, str, int]] = []
        self._current_context: Optional[Tuple[str, str, int]] = None
        self.datasets: List[Tuple[str, Any]] = []
        self.experiments: List[str] = []
        self.experiment_groups: List[Any] = []
        self.data_managers: Dict[str, Any] = {}
        self.metric_config = None
        self.evaluator_registry = None
        self.stored_plots: List[Tuple[str, Any]] = []
        self.stored_tables: List[Tuple[str, Any, Dict]] = []
        self.tuned_params: Dict[str, Any] = {}
        self.tuning_measure = None

    def set_metric_config(self, metric_config: Any):
        """Set metric configuration."""
        self.metric_config = metric_config

    def set_evaluator_registry(self, registry: Any):
        """Set evaluator registry."""
        self.evaluator_registry = registry

    def set_context(self, group_name: str, dataset_name: str, split_index: int):
        """Set current reporting context."""
        context = (group_name, dataset_name, split_index)
        self.contexts.append(context)
        self._current_context = context

    def get_context(self) -> Optional[Tuple[str, str, int]]:
        """Get current reporting context."""
        return self._current_context

    def clear_context(self):
        """Clear current context."""
        self._current_context = None

    def add_data_manager(self, group_name: str, data_manager: Any):
        """Add data manager to report."""
        self.data_managers[group_name] = data_manager

    def add_dataset(self, group_name: str, data_splits: Any):
        """Add dataset to report."""
        self.datasets.append((group_name, data_splits))

    def add_experiment(self, experiment_name: str):
        """Add experiment to report."""
        self.experiments.append(experiment_name)

    def add_experiment_groups(self, groups: List[Any]):
        """Add experiment groups."""
        self.experiment_groups.extend(groups)

    def store_plot_svg(self, plot_name: str, svg_data: Any, metadata: Dict):
        """Store plot SVG data."""
        self.stored_plots.append((plot_name, svg_data))

    def store_table_data(self, table_name: str, data: Any, metadata: Dict):
        """Store table data."""
        self.stored_tables.append((table_name, data, metadata))

    def cache_tuned_params(self, tuned_params: Dict[str, Any]):
        """Cache tuned hyperparameters."""
        self.tuned_params.update(tuned_params)

    def set_tuning_measure(self, measure: str):
        """Set the tuning measure."""
        self.tuning_measure = measure

    def get_report_data(self):
        """Get report data (mock)."""
        from unittest.mock import Mock
        return Mock(
            datasets=self.datasets,
            experiments=self.experiments,
            experiment_groups=self.experiment_groups,
            data_managers=self.data_managers
        )

    def clear(self):
        """Clear all report data."""
        self.contexts.clear()
        self._current_context = None
        self.datasets.clear()
        self.experiments.clear()
        self.experiment_groups.clear()
        self.data_managers.clear()
        self.stored_plots.clear()
        self.stored_tables.clear()
        self.tuned_params.clear()


class MockUtilityService(MockBaseService):
    """Mock utility service for testing.

    Mimics the interface of brisk.services.utility.UtilityService.
    """

    def __init__(self, name: str = "utility"):
        super().__init__(name)
        self.plot_settings = None
        self.algorithm_config = None
        self.split_indices = None
        self.cv_splitter = None

    def set_plot_settings(self, plot_settings: Any):
        """Set plot settings."""
        self.plot_settings = plot_settings

    def get_plot_settings(self):
        """Get plot settings."""
        return self.plot_settings

    def set_split_indices(self, split_indices: Any):
        """Set split indices."""
        self.split_indices = split_indices

    def set_algorithm_config(self, algorithm_config: Any):
        """Set algorithm configuration."""
        self.algorithm_config = algorithm_config

    def get_algo_wrapper(self, algorithm_name: str):
        """Get algorithm wrapper (returns None in mock)."""
        return None

    def get_group_index(self, is_test: bool):
        """Get group index (returns None in mock)."""
        return None

    def get_cv_splitter(self, split_method: str, **kwargs):
        """Get CV splitter (returns None in mock)."""
        return None


class MockRerunService(MockBaseService):
    """Mock rerun service for testing.

    Mimics the interface of brisk.services.rerun.RerunService.
    """

    def __init__(self, name: str = "rerun"):
        super().__init__(name)
        self.config_data = {}
        self.handlers = []

    def add_base_data_manager(self, config: Dict[str, Any]):
        """Add base data manager config (no-op in mock)."""
        pass

    def add_configuration(self, configuration: Dict[str, Any]):
        """Add configuration (no-op in mock)."""
        pass

    def add_experiment_groups(self, groups: List[Dict[str, Any]]):
        """Add experiment groups (no-op in mock)."""
        pass

    def add_metric_config(self, metric_configs: List[Dict[str, Any]]):
        """Add metric config (no-op in mock)."""
        pass

    def add_algorithm_config(self, algorithm_wrappers: List[Dict[str, Any]]):
        """Add algorithm config (no-op in mock)."""
        pass

    def add_evaluators_config(self, evaluator_configs: List[Dict[str, Any]]):
        """Add evaluators config (no-op in mock)."""
        pass

    def add_workflow_file(self, workflow_name: str, class_name: str):
        """Add workflow file (no-op in mock)."""
        pass

    def collect_dataset_metadata(self, groups_json):
        pass

class MockServiceBundle:
    """Complete mock service bundle for testing.

    Provides all Brisk services as mocks in a single bundle.

    Example:
        services = MockServiceBundle()
        services.logger.info("test")
        services.io.save_plot(plot, "test.png")

        assert ("INFO", "test") in services.logger.messages
        assert len(services.io.saved_plots) == 1
    """

    def __init__(self):
        self.logger = MockLoggingService()
        self.io = MockIOService()
        self.metadata = MockMetadataService()
        self.reporting = MockReportingService()
        self.utility = MockUtilityService()
        self.rerun = MockRerunService()

        # Cross-register services
        services_dict = {
            "logger": self.logger,
            "io": self.io,
            "metadata": self.metadata,
            "reporting": self.reporting,
            "utility": self.utility,
            "rerun": self.rerun,
        }

        for service in services_dict.values():
            service.register_services(services_dict)

    def clear_all(self):
        """Clear all service data."""
        self.logger.clear()
        self.io.clear()
        self.metadata.clear()
        self.reporting.clear()


# ============================================================================
# Model Mocks
# ============================================================================

class MockModel:
    """Mock sklearn-style model for testing.

    Simulates a trained model without actually training.

    Example:
        model = MockModel(predictions=[0, 1, 0, 1])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
    """

    def __init__(
        self,
        predictions: Optional[List[float]] = None,
        score: float = 0.85,
        feature_importances: Optional[List[float]] = None
    ):
        """Initialize mock model.

        Args:
            predictions: List of predictions to return
            score: Score to return from score() method
            feature_importances: Feature importances (None for no support)
        """
        self.is_fitted = False
        self.predictions = predictions or [0.0, 1.0, 0.0, 1.0]
        self._score = score
        self.feature_importances_ = feature_importances
        self.n_features_in_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """Mock fit method."""
        self.is_fitted = True
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else len(X[0])

        # Create fake feature importances if not provided
        if self.feature_importances_ is None:
            self.feature_importances_ = np.random.random(self.n_features_in_)
            self.feature_importances_ /= self.feature_importances_.sum()

        # Create fake coefficients for linear models
        self.coef_ = np.random.randn(self.n_features_in_)
        self.intercept_ = np.random.randn()

        return self

    def predict(self, X):
        """Mock predict method."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)

        # Cycle through predictions if we don't have enough
        result = []
        for i in range(n_samples):
            result.append(self.predictions[i % len(self.predictions)])

        return np.array(result)

    def predict_proba(self, X):
        """Mock predict_proba for classifiers."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        # Return fake probabilities (0.5, 0.5 for binary classification)
        return np.full((n_samples, 2), 0.5)

    def score(self, X, y):
        """Mock score method."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._score

    def get_params(self, deep=True):
        """Mock get_params for sklearn compatibility."""
        return {"mock_param": "mock_value"}

    def set_params(self, **params):
        """Mock set_params for sklearn compatibility."""
        return self


class MockScaler:
    """Mock sklearn-style scaler for testing.

    Example:
        scaler = MockScaler()
        scaler.fit(X_train)
        X_scaled = scaler.transform(X_test)
    """

    def __init__(self):
        self.is_fitted = False
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        """Mock fit method."""
        self.is_fitted = True
        if isinstance(X, pd.DataFrame):
            self.mean_ = X.mean().values
            self.scale_ = X.std().values
        else:
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        """Mock transform (returns input unchanged)."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return X

    def fit_transform(self, X, y=None):
        """Mock fit_transform."""
        self.fit(X, y)
        return self.transform(X)


# ============================================================================
# Data Mocks
# ============================================================================

class MockDataSplit:
    """Mock DataSplit for testing.

    Example:
        split = MockDataSplit(X_train, X_test, y_train, y_test)
        X_train, X_test, y_train, y_test = split.get_train_test()
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        scaler: Optional[Any] = None
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scaler = scaler or MockScaler()
        self.split_index = 0

    def get_train_test(self):
        """Get train/test split."""
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_train(self):
        """Get training data."""
        return self.X_train, self.y_train

    def get_test(self):
        """Get test data."""
        return self.X_test, self.y_test


class MockDataSplits:
    """Mock DataSplits container for testing.

    Example:
        splits = MockDataSplits([split1, split2, split3])
        split = splits.get_split(0)
    """

    def __init__(self, splits: Optional[List[MockDataSplit]] = None):
        self.splits = splits or []
        self.n_splits = len(self.splits)

    def get_split(self, index: int) -> MockDataSplit:
        """Get split by index."""
        return self.splits[index]

    def __len__(self):
        return self.n_splits

    def __iter__(self):
        return iter(self.splits)


# ============================================================================
# Helper Functions
# ============================================================================

def create_mock_services() -> MockServiceBundle:
    """Create a complete mock service bundle.

    Convenience function for getting mock services.

    Returns:
        MockServiceBundle with all services
    """
    return MockServiceBundle()


def create_mock_model(
    predictions: Optional[List[float]] = None,
    score: float = 0.85
) -> MockModel:
    """Create a mock model with sensible defaults.

    Args:
        predictions: Predictions to return
        score: Score to return

    Returns:
        MockModel instance
    """
    return MockModel(predictions=predictions, score=score)


def create_mock_split(
    rows_train: int = 8,
    rows_test: int = 2,
    n_features: int = 2
) -> MockDataSplit:
    """Create a mock data split with synthetic data.

    Args:
        rows_train: Number of training samples
        rows_test: Number of test samples
        n_features: Number of features

    Returns:
        MockDataSplit instance
    """
    X_train = pd.DataFrame(
        np.random.randn(rows_train, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    X_test = pd.DataFrame(
        np.random.randn(rows_test, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    y_train = pd.Series(np.random.randint(0, 2, rows_train), name="target")
    y_test = pd.Series(np.random.randint(0, 2, rows_test), name="target")

    return MockDataSplit(X_train, X_test, y_train, y_test)
