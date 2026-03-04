"""MetricManager unit tests"""

import pytest

from brisk.evaluation import metric_manager
from brisk.evaluation import metric_wrapper
from brisk.defaults import regression_metrics, classification_metrics

def wrapper_factory(name="custom", abbr="cstm", display_name="Custom Wrapper"):

    def custom_calculation(y_true, y_pred):
        return sum(y_true) / sum(y_pred)


    return  metric_wrapper.MetricWrapper(
        name=name,
        abbr=abbr,
        display_name=display_name,
        func= custom_calculation,
        greater_is_better=False
    )


@pytest.mark.unit
class TestMetricManagerUnit:
    def test_init_no_wrapper(self):
        manager = metric_manager.MetricManager()
        assert isinstance(manager, metric_manager.MetricManager)

    def test_init_one_wrapper(self):
        manager = metric_manager.MetricManager(wrapper_factory())
        assert len(manager.list_metrics()) == 1

    def test_init_two_wrappers(self):
        manager = metric_manager.MetricManager(
            wrapper_factory(),
            wrapper_factory("second", "sec", "Second Wrapper")
        )
        assert len(manager.list_metrics()) == 2

    def test_init_duplicates_override(self):
        manager = metric_manager.MetricManager(
            wrapper_factory(),
            wrapper_factory("second", "sec", "Second Wrapper"),
            wrapper_factory()
        )
        assert len(manager.list_metrics()) == 2

    def test_get_metric_by_name(self):
        manager = metric_manager.MetricManager(
            *regression_metrics.REGRESSION_METRICS
        )
        function = manager.get_metric("mean_absolute_error")
        assert function.func.__name__ == "mean_absolute_error"

    def test_get_metric_by_abbreviation(self):
        manager = metric_manager.MetricManager(
            *regression_metrics.REGRESSION_METRICS
        )
        function = manager.get_metric("MSE")
        assert function.func.__name__ == "mean_squared_error"

    def test_get_metric_by_display_name(self):
        manager = metric_manager.MetricManager(
            *regression_metrics.REGRESSION_METRICS
        )
        function = manager.get_metric("Root Mean Squared Error")
        assert function.func.__name__ == "root_mean_squared_error"

    def test_get_metric_missing(self):
        manager = metric_manager.MetricManager(
            *regression_metrics.REGRESSION_METRICS
        )
        with pytest.raises(ValueError):
            _ = manager.get_metric("accuracy")

    def test_export_config_default_regression(self):
        manager = metric_manager.MetricManager(
            *regression_metrics.REGRESSION_METRICS
        )
        export = manager.export_params()
        assert export == [{
            "type": "builtin_collection",
            "collection": "brisk.REGRESSION_METRICS"
        }]

    def test_export_config_default_classification(self):
        manager = metric_manager.MetricManager(
            *classification_metrics.CLASSIFICATION_METRICS
        )
        export = manager.export_params()
        assert export == [{
            "type": "builtin_collection",
            "collection": "brisk.CLASSIFICATION_METRICS"
        }]

    def test_export_params_custom_metrics(self):
        manager = metric_manager.MetricManager(wrapper_factory())
        export = manager.export_params()
        assert export[0]["type"] == "custom_metric"
        assert export[0]["name"] == "custom"
        assert export[0]["func_type"] == "local"
