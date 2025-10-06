"""Custom assertions for better test error messages.

This module provides custom assertion functions that give more helpful error
messages than standard assertions, making it easier to debug failing tests.

Usage:
    from tests.utils.assertions import assert_dataframe_equal, assert_files_exist
    
    assert_dataframe_equal(actual_df, expected_df)
    assert_files_exist(output_dir, ["model.pkl", "metrics.json"])
"""
from typing import Any, List, Dict, Union
from pathlib import Path

import pandas as pd
import numpy as np


def assert_dataframe_equal(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    check_dtype: bool = True,
    check_names: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8
):
    """Assert two DataFrames are equal with enhanced error messages.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        check_dtype: Whether to check dtype equality
        check_names: Whether to check column/index names
        rtol: Relative tolerance for numeric comparison
        atol: Absolute tolerance for numeric comparison
        
    Raises:
        AssertionError: If DataFrames are not equal, with detailed context
    """
    try:
        pd.testing.assert_frame_equal(
            df1, df2,
            check_dtype=check_dtype,
            check_names=check_names,
            rtol=rtol,
            atol=atol
        )
    except AssertionError as e:
        msg = ["\nDataFrames are not equal:"]
        msg.append(f"  Shape df1: {df1.shape}")
        msg.append(f"  Shape df2: {df2.shape}")

        if df1.shape != df2.shape:
            msg.append("  ❌ Shapes differ!")

        msg.append(f"  Columns df1: {list(df1.columns)}")
        msg.append(f"  Columns df2: {list(df2.columns)}")

        if list(df1.columns) != list(df2.columns):
            msg.append("  ❌ Columns differ!")
            missing_in_df2 = set(df1.columns) - set(df2.columns)
            missing_in_df1 = set(df2.columns) - set(df1.columns)
            if missing_in_df2:
                msg.append(f"  Missing in df2: {missing_in_df2}")
            if missing_in_df1:
                msg.append(f"  Missing in df1: {missing_in_df1}")

        if df1.shape == df2.shape and list(df1.columns) == list(df2.columns):
            try:
                diff = (df1 != df2).sum().sum()
                msg.append(f"  Different values: {diff} cells")
            except (ValueError, TypeError, AttributeError):
                pass

        msg.append(f"\nOriginal pandas error:\n{str(e)}")
        raise AssertionError("\n".join(msg)) # pylint: disable=raise-missing-from


def assert_files_exist(directory: Union[str, Path], expected_files: List[str]):
    """Assert that all expected files exist in directory.
    
    Args:
        directory: Directory to check
        expected_files: List of filenames that should exist
        
    Raises:
        AssertionError: If any files are missing
    """
    directory = Path(directory)

    if not directory.exists():
        raise AssertionError(f"Directory does not exist: {directory}")

    missing_files = []
    for file in expected_files:
        file_path = directory / file
        if not file_path.exists():
            missing_files.append(file)

    if missing_files:
        msg = [f"\nMissing files in {directory}:"]
        for f in missing_files:
            msg.append(f"  ❌ {f}")

        # Show what files DO exist
        existing = [f.name for f in directory.iterdir() if f.is_file()]
        if existing:
            msg.append("\nFiles that exist:")
            for f in existing:
                msg.append(f"  ✓ {f}")

        raise AssertionError("\n".join(msg))


def assert_dict_contains(
    actual: Dict,
    expected: Dict,
    check_values: bool = True
):
    """Assert that actual dict contains all keys/values from expected dict.
    
    Args:
        actual: Actual dictionary
        expected: Expected dictionary (subset)
        check_values: Whether to check values (if False, only checks keys)
        
    Raises:
        AssertionError: If expected keys/values are not in actual dict
    """
    missing_keys = set(expected.keys()) - set(actual.keys())
    if missing_keys:
        msg = ["\nMissing keys in dict:"]
        for key in missing_keys:
            msg.append(f"  ❌ {key}")

        msg.append(f"\nActual keys: {list(actual.keys())}")
        msg.append(f"Expected keys: {list(expected.keys())}")
        raise AssertionError("\n".join(msg))

    if check_values:
        mismatched_values = {}
        for key, expected_value in expected.items():
            actual_value = actual[key]

            # Handle different types of comparisons
            if isinstance(expected_value, (int, float, str, bool)):
                if actual_value != expected_value:
                    mismatched_values[key] = {
                        "expected": expected_value,
                        "actual": actual_value
                    }
            elif isinstance(expected_value, (list, tuple)):
                if list(actual_value) != list(expected_value):
                    mismatched_values[key] = {
                        "expected": expected_value,
                        "actual": actual_value
                    }

        if mismatched_values:
            msg = ["\nDict values don't match:"]
            for key, values in mismatched_values.items():
                msg.append(f"  {key}:")
                msg.append(f"    Expected: {values["expected"]}")
                msg.append(f"    Actual:   {values["actual"]}")
            raise AssertionError("\n".join(msg))


def assert_dict_equal(dict1: Dict, dict2: Dict):
    """Assert two dictionaries are exactly equal.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Raises:
        AssertionError: If dictionaries differ
    """
    if dict1 == dict2:
        return

    msg = ["\nDictionaries are not equal:"]

    # Check keys
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    missing_in_dict2 = keys1 - keys2
    missing_in_dict1 = keys2 - keys1
    common_keys = keys1 & keys2

    if missing_in_dict2:
        msg.append(f"  Keys in dict1 but not dict2: {missing_in_dict2}")

    if missing_in_dict1:
        msg.append(f"  Keys in dict2 but not dict1: {missing_in_dict1}")

    # Check values for common keys
    different_values = []
    for key in common_keys:
        if dict1[key] != dict2[key]:
            different_values.append(key)

    if different_values:
        msg.append("  Different values for keys:")
        for key in different_values:
            msg.append(f"    {key}:")
            msg.append(f"      dict1: {dict1[key]}")
            msg.append(f"      dict2: {dict2[key]}")

    raise AssertionError("\n".join(msg))


def assert_model_fitted(model: Any):
    """Assert that a sklearn model has been fitted.
    
    Args:
        model: sklearn-style model
        
    Raises:
        AssertionError: If model does not appear to be fitted
    """
    # Check common fitted attributes
    fitted_attrs = [
        "coef_", "intercept_", "n_features_in_",
        "feature_importances_", "classes_", "tree_"
    ]

    has_fitted_attr = any(hasattr(model, attr) for attr in fitted_attrs)

    if not has_fitted_attr:
        model_attrs = [attr for attr in dir(model) if not attr.startswith("_")]
        raise AssertionError(
            f"Model {type(model).__name__} does not appear to be fitted.\n"
            f"No common fitted attributes found.\n"
            f"Expected one of: {fitted_attrs}\n"
            f"Available attributes: {model_attrs[:10]}..."  # Show first 10
        )


def assert_in_range(
    value: float,
    min_val: float,
    max_val: float,
    inclusive: bool = True
):
    """Assert that a value is within a range.
    
    Args:
        value: Value to check
        min_val: Minimum value
        max_val: Maximum value
        inclusive: Whether range is inclusive
        
    Raises:
        AssertionError: If value is out of range
    """
    if inclusive:
        in_range = min_val <= value <= max_val
        comparison = f"[{min_val}, {max_val}]"
    else:
        in_range = min_val < value < max_val
        comparison = f"({min_val}, {max_val})"

    if not in_range:
        raise AssertionError(
            f"Value {value} is not in range {comparison}"
        )


def assert_has_attributes(obj: Any, attributes: List[str]):
    """Assert that an object has all specified attributes.
    
    Args:
        obj: Object to check
        attributes: List of attribute names that should exist
        
    Raises:
        AssertionError: If any attributes are missing
    """
    missing_attrs = [attr for attr in attributes if not hasattr(obj, attr)]

    if missing_attrs:
        available_attrs = [
            attr for attr in dir(obj)
            if not attr.startswith("_")
        ]
        raise AssertionError(
            f"Object {type(obj).__name__} is missing attributes:\n"
            f"  Missing: {missing_attrs}\n"
            f"  Available: {available_attrs[:10]}..."
        )


def assert_is_instance_of(obj: Any, expected_types: Union[type, tuple]):
    """Assert object is instance of expected type(s) with helpful message.
    
    Args:
        obj: Object to check
        expected_types: Expected type or tuple of types
        
    Raises:
        AssertionError: If object is not an instance of expected type(s)
    """
    if not isinstance(obj, expected_types):
        if isinstance(expected_types, tuple):
            type_names = " or ".join(t.__name__ for t in expected_types)
        else:
            type_names = expected_types.__name__

        raise AssertionError(
            f"Expected instance of {type_names}, "
            f"but got {type(obj).__name__}"
        )


def assert_shape_equal(arr1: np.ndarray, arr2: np.ndarray):
    """Assert two arrays have the same shape.
    
    Args:
        arr1: First array
        arr2: Second array
        
    Raises:
        AssertionError: If shapes don't match
    """
    if arr1.shape != arr2.shape:
        raise AssertionError(
            f"Array shapes don't match:\n"
            f"  arr1: {arr1.shape}\n"
            f"  arr2: {arr2.shape}"
        )


def assert_no_missing_values(df: pd.DataFrame):
    """Assert DataFrame has no missing values.
    
    Args:
        df: DataFrame to check
        
    Raises:
        AssertionError: If DataFrame contains missing values
    """
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]

        msg = ["\nDataFrame contains missing values:"]
        for col, count in cols_with_nulls.items():
            msg.append(f"  {col}: {count} missing values")

        raise AssertionError("\n".join(msg))


def assert_columns_equal(df1: pd.DataFrame, df2: pd.DataFrame):
    """Assert two DataFrames have the same columns.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        
    Raises:
        AssertionError: If columns don't match
    """
    if list(df1.columns) != list(df2.columns):
        missing_in_df2 = set(df1.columns) - set(df2.columns)
        missing_in_df1 = set(df2.columns) - set(df1.columns)

        msg = ["\nDataFrame columns don't match:"]
        if missing_in_df2:
            msg.append(f"  In df1 but not df2: {missing_in_df2}")
        if missing_in_df1:
            msg.append(f"  In df2 but not df1: {missing_in_df1}")
        msg.append(f"  df1 columns: {list(df1.columns)}")
        msg.append(f"  df2 columns: {list(df2.columns)}")

        raise AssertionError("\n".join(msg))
