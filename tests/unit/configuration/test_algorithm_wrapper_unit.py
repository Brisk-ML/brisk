"""Unit tests for AlgorithmWrapper."""

import pytest
from sklearn import linear_model, base

from brisk import AlgorithmWrapper

# pylint: disable=W0612

@pytest.mark.unit
class TestAlgorithmWrapperUnit():
    """Unit tests for the AlgorithmWrapper class."""

    def test_instantiate_no_optional(self):
        wrapper = AlgorithmWrapper(
            name="test_wrapper",
            display_name="Test Wrapper",
            algorithm_class = linear_model.Ridge
        )
        assert wrapper.default_params == {}
        assert wrapper.hyperparam_grid == {}

    def test_instantiate_all_optionals(self):
        default_params = {"max_iter": 1000}
        hyperparam_grid = {"alpha": [0.1, 0.5, 1.0]}
        wrapper = AlgorithmWrapper(
            name="test_wrapper",
            display_name="Test Wrapper",
            algorithm_class = linear_model.Ridge,
            default_params=default_params,
            hyperparam_grid=hyperparam_grid
        )
        assert wrapper.default_params == default_params
        assert wrapper.hyperparam_grid == hyperparam_grid

    def test_instantiate_non_sklearn_class(self):
        class NonSklearnClass():
            pass

        with pytest.raises(ValueError):
            wrapper = AlgorithmWrapper(
                name="test_wrapper",
                display_name="Test Wrapper",
                algorithm_class = NonSklearnClass
            )

    def test_instantiate_base_learner_subclass(self):
        class InheritBaseEstimator(base.BaseEstimator):
            pass

        wrapper = AlgorithmWrapper(
            name="test_wrapper",
            display_name="Test Wrapper",
            algorithm_class = InheritBaseEstimator
        )
        assert wrapper.algorithm_class == InheritBaseEstimator

    def test_instantiate_default_params_type_error(self):
        with pytest.raises(TypeError):
            wrapper = AlgorithmWrapper(
                name="test_wrapper",
                display_name="Test Wrapper",
                algorithm_class = linear_model.Ridge,
                default_params=[0.1, 0.5, 1.0]
            )

    def test_instantiate_hyperparam_grid_type_error(self):
        with pytest.raises(TypeError):
            wrapper = AlgorithmWrapper(
                name="test_wrapper",
                display_name="Test Wrapper",
                algorithm_class = linear_model.Ridge,
                hyperparam_grid=[0.1, 0.5, 1.0]
            )

    def test_set_default_params(self):
        wrapper = AlgorithmWrapper(
            name="test_wrapper",
            display_name="Test Wrapper",
            algorithm_class = linear_model.Ridge
        )
        assert wrapper.default_params == {}
        default_params = {"max_iter": 1000, "alpha": 0.6}
        wrapper["default_params"] = default_params
        assert wrapper.default_params == default_params

    def test_set_hyperparam_grid(self):
        wrapper = AlgorithmWrapper(
            name="test_wrapper",
            display_name="Test Wrapper",
            algorithm_class = linear_model.Ridge
        )
        assert wrapper.hyperparam_grid == {}
        hyperparam_grid = {"alpha": [0.6, 0.65, 0.7]}
        wrapper["hyperparam_grid"] = hyperparam_grid
        assert wrapper.hyperparam_grid == hyperparam_grid

    def test_set_invalid_key(self):
        wrapper = AlgorithmWrapper(
            name="test_wrapper",
            display_name="Test Wrapper",
            algorithm_class = linear_model.Ridge
        )
        with pytest.raises(KeyError):
            wrapper["invalid_key"] = {"test": "error"}

    def test_update_existing_dict_key(self):
        wrapper = AlgorithmWrapper(
            name="test_wrapper",
            display_name="Test Wrapper",
            algorithm_class = linear_model.Ridge,
            default_params = {"max_iter": 150}
        )
        new_default_params = {"alpha": 0.75, "max_iter": 10000}
        wrapper["default_params"] = new_default_params
        assert wrapper.default_params == new_default_params

    def test_instantiate_has_correct_params(self):
        wrapper = AlgorithmWrapper(
            name="test_wrapper",
            display_name="Test Wrapper",
            algorithm_class = linear_model.Ridge,
            default_params = {"max_iter": 150}
        )
        instantiated_model = wrapper.instantiate()
        assert instantiated_model.max_iter == 150

    def test_instantiate_has_wrapper_name(self):
        wrapper = AlgorithmWrapper(
            name="test_wrapper",
            display_name="Test Wrapper",
            algorithm_class = linear_model.Ridge,
            default_params = {"max_iter": 150}
        )
        instantiated_model = wrapper.instantiate()
        assert instantiated_model.wrapper_name == "test_wrapper"

    @pytest.mark.parametrize(
        "default_params, best_params",
        [
            pytest.param({"alpha": 0.2}, {}),
            pytest.param({}, {}),
            pytest.param({}, {"alpha": 0.5}),
            pytest.param({"alpha": 0.2}, {"alpha": 0.5}),
        ],
    )
    def test_instantiate_tuned(self, default_params, best_params):
        wrapper = AlgorithmWrapper(
            name="test_wrapper",
            display_name="Test Wrapper",
            algorithm_class = linear_model.Ridge,
            default_params = default_params
        )
        instantiated_model = wrapper.instantiate_tuned(best_params)
        correct_value = default_params.copy()
        correct_value.update(best_params)
        assert instantiated_model.alpha == correct_value.get("alpha", 1.0)

    def test_instantiate_tuned_type_error(self):
        wrapper = AlgorithmWrapper(
            name="test_wrapper",
            display_name="Test Wrapper",
            algorithm_class = linear_model.Ridge,
        )
        with pytest.raises(TypeError):
            wrapper.instantiate_tuned([0.1, 0.5, 1.0])

    def test_instantiate_tuned_has_wrapper_name(self):
        wrapper = AlgorithmWrapper(
            name="test_wrapper",
            display_name="Test Wrapper",
            algorithm_class = linear_model.Ridge,
        )
        tuned_model = wrapper.instantiate_tuned({"alpha": 0.1})
        assert tuned_model.wrapper_name == "test_wrapper"
