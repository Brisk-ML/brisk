"""Test utilities for the Brisk test suite.

This package provides factories, mocks, and custom assertions to make
writing tests easier and more maintainable.

Quick Start:
    # Factories - create test data with flexible parameters
    from tests.utils.factories import DataFrameFactory, AlgorithmFactory

    df = DataFrameFactory.simple(rows=100)
    algorithms = AlgorithmFactory.collection(n=3)

    # Mocks - fake objects for isolated unit tests
    from tests.utils.mocks import MockServiceBundle, MockModel

    services = MockServiceBundle()
    model = MockModel(predictions=[0, 1, 0, 1])

    # Assertions - better error messages
    from tests.utils.assertions import assert_dataframe_equal, assert_files_exist

    assert_dataframe_equal(actual_df, expected_df)
    assert_files_exist(output_dir, ["model.pkl", "metrics.json"])
"""

# Expose commonly used factories
from tests.utils.factories import (
    DataFrameFactory,
    AlgorithmFactory,
    MetricFactory,
    DataManagerFactory,
    ExperimentGroupFactory,
    ConfigurationFactory,
)

# Expose commonly used mocks
from tests.utils.mocks import (
    MockServiceBundle,
    MockModel,
    MockScaler,
    MockDataSplit,
    MockDataSplits,
    create_mock_services,
    create_mock_model,
    create_mock_split,
)

# Expose commonly used assertions
from tests.utils.assertions import (
    assert_dataframe_equal,
    assert_files_exist,
    assert_dict_contains,
    assert_dict_equal,
    assert_model_fitted,
    assert_in_range,
    assert_has_attributes,
    assert_is_instance_of,
    assert_no_missing_values,
    assert_columns_equal,
)

__all__ = [
    # Factories
    "DataFrameFactory",
    "AlgorithmFactory",
    "MetricFactory",
    "DataManagerFactory",
    "ExperimentGroupFactory",
    "ConfigurationFactory",
    # Mocks
    "MockServiceBundle",
    "MockModel",
    "MockScaler",
    "MockDataSplit",
    "MockDataSplits",
    "create_mock_services",
    "create_mock_model",
    "create_mock_split",
    # Assertions
    "assert_dataframe_equal",
    "assert_files_exist",
    "assert_dict_contains",
    "assert_dict_equal",
    "assert_model_fitted",
    "assert_in_range",
    "assert_has_attributes",
    "assert_is_instance_of",
    "assert_no_missing_values",
    "assert_columns_equal",
]

