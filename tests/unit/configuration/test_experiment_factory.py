"""Unit tests for ExperimentFactory"""
import pytest

from brisk.configuration.project import ProjectRootContext
from brisk.configuration.experiment_factory import ExperimentFactory

from tests.utils.factories import AlgorithmFactory, ExperimentGroupFactory

@pytest.fixture
def algo_collection():
    return AlgorithmFactory.collection()


class TestExperimentFactory:
    def test_initalization(self, algo_collection):
        categorical_features = {"data.csv": ["country"]}
        factory = ExperimentFactory(algo_collection, categorical_features)
        assert factory.algorithm_config == algo_collection
        assert factory.categorical_features == categorical_features

    def test_init_non_algorithm_collection(self, algo_collection):
        with pytest.raises(TypeError):
            factory = ExperimentFactory(list(algo_collection), {})

    def test_create_experiments_0_splits(self, tmp_path, algo_collection):
        factory = ExperimentFactory(algo_collection, {})
        with ProjectRootContext(tmp_path):
            group = ExperimentGroupFactory.simple(tmp_path)
            experiments = factory.create_experiments(group, 0)
        assert len(experiments) == 0

    def test_create_experiments_1_split(self, tmp_path, algo_collection):
        factory = ExperimentFactory(algo_collection, {})
        with ProjectRootContext(tmp_path):
            group = ExperimentGroupFactory.simple(tmp_path)
            experiments = factory.create_experiments(group, 1)
        assert len(experiments) == 1

    def test_create_experiment_2_splits(self, tmp_path, algo_collection):
        factory = ExperimentFactory(algo_collection, {})
        with ProjectRootContext(tmp_path):
            group = ExperimentGroupFactory.simple(
                tmp_path, algorithms=["ridge", "lasso"]
            )
            experiments = factory.create_experiments(group, 2)
        assert len(experiments) == 4

    def test_create_experiments_group_hyperparam_grid_updated(
        self,
        tmp_path,
        algo_collection
    ):
        """ExperimentGroup algo_config should update the wrapper grid."""
        hyperparam_grid = {"ridge": {"alpha": [0, 0.1, 0.2, 0.3, 0.4]}}
        factory = ExperimentFactory(algo_collection, {})
        with ProjectRootContext(tmp_path):
            group = ExperimentGroupFactory.with_hyperparam_grid(
                tmp_path, hyperparam_grid 
            )
            experiments = factory.create_experiments(group, 1)

        assert len(experiments) == 1
        exp = experiments[0]
        assert exp.algorithms["model"].hyperparam_grid == {
            "alpha": [0, 0.1, 0.2, 0.3, 0.4]
        }

    def test_two_algorithms_use_correct_model_keys(
        self,
        tmp_path,
        algo_collection
    ):
        """If one experiment has two algorithms use correct keys."""
        factory = ExperimentFactory(algo_collection, {})
        with ProjectRootContext(tmp_path):
            group = ExperimentGroupFactory.simple(tmp_path, algorithms=[[
                "ridge", "linear"
            ]])
            experiments = factory.create_experiments(group, 1)
        assert len(experiments) == 1
        exp = experiments[0]
        assert list(exp.algorithm_kwargs.keys()) == ["model", "model2"]

    def test_missing_algorithm_wrapper(
        self,
        tmp_path,
        algo_collection
    ):
        factory = ExperimentFactory(algo_collection, {})
        with ProjectRootContext(tmp_path):
            group = ExperimentGroupFactory.simple(
                tmp_path, algorithms=["rf"]
            )
            with pytest.raises(KeyError):
                experiments = factory.create_experiments(group, 1)
    
