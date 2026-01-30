"""Unit tests for MetricWrapper."""

import functools
from typing import Callable

import pytest

from brisk.evaluation import metric_wrapper


def custom_metric(y_true, y_pred):
    pass


def another_custom_metric(y_true, y_pred, param1, split_metadata):
    pass


def a_third_metric(y_true, y_pred, param1, param2):
    pass


@pytest.mark.unit
class TestMetricWrapperUnit:
    def test_initalization(self):
        wrapper = metric_wrapper.MetricWrapper(
            name="custom",
            func=custom_metric,
            display_name="Custom Metric",
            greater_is_better=True,
            abbr="cst"
        )
        assert isinstance(wrapper, metric_wrapper.MetricWrapper)

    def test_initalzation_no_optional(self):
        wrapper = metric_wrapper.MetricWrapper(
            name="custom",
            func=custom_metric,
            display_name="Custom Metric",
            greater_is_better=True,
        )
        assert isinstance(wrapper, metric_wrapper.MetricWrapper)
        assert wrapper.abbr == "custom"

    def test_apply_params_no_params(self):
        wrapper = metric_wrapper.MetricWrapper(
            name="custom",
            func=custom_metric,
            display_name="Custom Metric",
            greater_is_better=True,
            abbr="cst"
        )
        assert isinstance(wrapper.scorer, Callable)
        assert isinstance(wrapper._func_with_params, functools.partial)
        assert wrapper._func_with_params.__name__ == "custom"

    def test_apply_params_one_param(self):
        wrapper = metric_wrapper.MetricWrapper(
            name="another_custom",
            func=another_custom_metric,
            display_name="Custom Metric",
            greater_is_better=True,
            param1=5
        )
        assert "param1" in wrapper.params
        assert wrapper.params["param1"] == 5
        expected_kwargs = {"param1": 5, "split_metadata": {}}
        assert wrapper._func_with_params.keywords == expected_kwargs

    def test_apply_params_two_params(self):
        param1 = 10
        param2 = "some string"
        wrapper = metric_wrapper.MetricWrapper(
            name="another_custom",
            func=a_third_metric,
            display_name="Custom Metric",
            greater_is_better=True,
            param1=param1,
            param2=param2
        )
        assert "param1" in wrapper.params
        assert wrapper.params["param1"] == param1
        assert "param2" in wrapper.params
        assert wrapper.params["param2"] == param2
        expected_kwargs = {
            "param1": param1,
            "param2": param2,
            "split_metadata": {}
        }
        assert wrapper._func_with_params.keywords == expected_kwargs

    def test_set_params_updates_partial(self):
        param1 = 7
        wrapper = metric_wrapper.MetricWrapper(
            name="another_custom",
            func=another_custom_metric,
            display_name="Another Custom",
            greater_is_better=True,
            param1=param1
        )
        wrapper.set_params(param1=20)
        assert wrapper.params["param1"] == 20
        assert wrapper._func_with_params.keywords["param1"] == 20

    def test_get_func_with_params_deepcopy(self):
        wrapper = metric_wrapper.MetricWrapper(
            name="custom",
            func=custom_metric,
            display_name="Custom Metric",
            greater_is_better=True,
        )

        func1 = wrapper.get_func_with_params()
        func2 = wrapper.get_func_with_params()

        assert func1 is not func2
        assert func1 is not wrapper._func_with_params

    def test_ensure_split_metadata_param_is_missing(self):
        wrapper = metric_wrapper.MetricWrapper(
            name="custom",
            func=custom_metric,
            display_name="Custom Metric",
            greater_is_better=True,
            abbr="cst"
        )
        assert "split_metadata" in wrapper.params
        assert wrapper._func_with_params.keywords == {"split_metadata": {}}
