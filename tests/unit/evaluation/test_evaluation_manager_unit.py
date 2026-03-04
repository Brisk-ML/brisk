"""Unit tests for EvaluationManager."""

import pathlib

import pytest

from brisk.evaluation.evaluation_manager import EvaluationManager

from tests.utils.factories import MetricManagerFactory

@pytest.fixture()
def metric_manager():
    return MetricManagerFactory.regression()


@pytest.fixture()
def eval_manager(metric_manager):
    return EvaluationManager(metric_manager)


@pytest.mark.unit
class TestEvaluationManagerUnit:
    def test_init_metric_deep_copy(self, metric_manager):
        evaluation_manager = EvaluationManager(metric_manager)
        assert evaluation_manager.metric_manager is not metric_manager

    def test_set_output_dir_makes_path(self, eval_manager):
        path = "path/to/output"
        eval_manager.set_output_dir(path)
        assert isinstance(eval_manager.output_dir, pathlib.Path)
