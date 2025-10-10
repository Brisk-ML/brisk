"""Unit tests for AlgorithmWrapper."""
import pytest
from sklearn import linear_model
from sklearn.base import BaseEstimator

from brisk.configuration.algorithm_wrapper import AlgorithmWrapper
from utils.factories import AlgorithmFactory

# Disable redefined-outer-name to prevent fixtures from raising lint errors
# pylint: disable = W0621

@pytest.fixture
def linear_wrapper():
    """LinearRegression AlgorithmWrapper instance"""
    return AlgorithmFactory.linear()


@pytest.fixture
def ridge_wrapper():
    """Ridge AlgorithmWrapper instance"""
    return AlgorithmFactory.ridge(with_hyperparams=True)


@pytest.fixture
def rf_wrapper():
    """LinearRegression AlgorithmWrapper instance"""
    return AlgorithmFactory.random_forest(with_hyperparams=True)


class TestAlgorithmWrapper:
    """Test class for AlgorithmWrapper."""
    def test_init(self, linear_wrapper):
        """Test the initialization of AlgorithmWrapper."""
        assert linear_wrapper.name == "linear"
        assert linear_wrapper.display_name == "Linear Regression"
        assert linear_wrapper.algorithm_class == linear_model.LinearRegression
        assert linear_wrapper.default_params == {}
        assert linear_wrapper.hyperparam_grid == {}

    def test_init_with_all_params(self, ridge_wrapper):
        """Test the initialization of AlgorithmWrapper with all parameters."""
        assert ridge_wrapper.name == "ridge"
        assert ridge_wrapper.display_name == "Ridge Regression"
        assert ridge_wrapper.algorithm_class == linear_model.Ridge
        assert ridge_wrapper.default_params == {
            "alpha": 1.0,
            "max_iter": 10000,
        }
        assert ridge_wrapper.hyperparam_grid == {"alpha": [0.1, 0.5, 1.0]}

    def test_init_int_name_raise_error(self):
        """Test init error handling; name passed int type."""
        with pytest.raises(TypeError, match="name must be a string"):
            AlgorithmWrapper(
                name=123,
                display_name="Linear Regression",
                algorithm_class=linear_model.LinearRegression,
            )

    def test_init_list_name_raise_error(self):
        """Test init error handling; name passed list type."""
        with pytest.raises(TypeError, match="name must be a string"):
            AlgorithmWrapper(
                name=["linear"],
                display_name="Linear Regression",
                algorithm_class=linear_model.LinearRegression,
            )

    def test_init_float_display_name_error(self):
        """Test init error handling; display_name passed float type."""
        with pytest.raises(TypeError, match="display_name must be a string"):
            AlgorithmWrapper(
                name="linear",
                display_name=123.123,
                algorithm_class=linear_model.LinearRegression,
            )

    def test_init_dict_display_name_error(self):
        """Test init error handling; name passed dict type."""
        with pytest.raises(TypeError, match="display_name must be a string"):
            AlgorithmWrapper(
                name="linear",
                display_name={"display_name": "linear"},
                algorithm_class=linear_model.LinearRegression,
            )

    def test_init_str_algorithm_class_error(self):
        """Test init error handling; algorithm_class passed string type."""
        with pytest.raises(
            TypeError,
            match="algorithm_class must be a class"
        ):
            AlgorithmWrapper(
                name="linear",
                display_name="Linear Regression",
                algorithm_class="linear",
            )

    def test_init_algorithm_class_error(self):
        """Test init error handling; algorithm_class passed MockClass object."""
        class MockClass:
            pass

        with pytest.raises(
            ValueError, match="must be a subclass of sklearn.base.BaseEstimator"
            ):
            AlgorithmWrapper(
                name="linear",
                display_name="Linear Regression",
                algorithm_class=MockClass,
            )

    def test_init_algorithm_class_accepts_base_estimator(self):
        """Test algorithm_class accepts BaseEstimator subclass.

        If this test fails it means AlgorithmWrapper is not detecting
        BaseEstimator subclasses properly.
        """
        class MockClass(BaseEstimator):
            pass

        AlgorithmWrapper(
            name="linear",
            display_name="Linear Regression",
            algorithm_class=MockClass,
        )

    def test_init_str_default_params_error(self):
        """Test init error handling; default_params passed str type."""
        with pytest.raises(
            TypeError, match="default_params must be a dictionary"
            ):
            AlgorithmWrapper(
                name="linear",
                display_name="Linear Regression",
                algorithm_class=linear_model.LinearRegression,
                default_params="fit_intercept",
            )

    def test_init_str_hyperparm_grid_error(self):
        """Test init error handling; hyperparam_grid passed str type."""
        with pytest.raises(
            TypeError, match="hyperparam_grid must be a dictionary"
            ):
            AlgorithmWrapper(
                name="linear",
                display_name="Linear Regression",
                algorithm_class=linear_model.LinearRegression,
                hyperparam_grid="alpha: [0.01, 0.1, 1.0]",
            )

    def test_setitem_default_params(self, linear_wrapper):
        """Test __setitem__ method accepts default_params key."""
        assert linear_wrapper.default_params == {}
        linear_wrapper["default_params"] = {"param1": 5, "param2": "test"}
        assert linear_wrapper.default_params["param1"] == 5
        assert linear_wrapper.default_params["param2"] == "test"

    def test_setitem_hyperparam_grid(self, linear_wrapper):
        """Test __setitem__ method accepts hyperparam_grid key."""
        assert linear_wrapper.hyperparam_grid == {}
        linear_wrapper["hyperparam_grid"] = {"param1": [1, 2, 3]}
        assert linear_wrapper.hyperparam_grid["param1"] == [1, 2, 3]

    def test_setitem_invalid_key(self, linear_wrapper):
        """Test __setitem__ method raises KeyError for invalid keys."""
        with pytest.raises(KeyError, match="Invalid key: invalid_key"):
            linear_wrapper["invalid_key"] = {"param1": 10}

    def test_setitem_bool_invalid_value(self, linear_wrapper):
        """Test __setitem__ method raises TypeError for bool type."""
        with pytest.raises(TypeError, match="value must be a dict"):
            linear_wrapper["default_params"] = True

    def test_setitem_str_invalid_value(self, linear_wrapper):
        """Test __setitem__ method raises TypeError for str type."""
        with pytest.raises(TypeError, match="value must be a dict"):
            linear_wrapper["hyperparam_grid"] = "param"

    def test_setitem_list_invalid_value(self, linear_wrapper):
        """Test __setitem__ method raises TypeError for list type."""
        with pytest.raises(TypeError, match="value must be a dict"):
            linear_wrapper["hyperparam_grid"] = ["can't", "be", "a", "list"]

    def test_instantiate_no_default_params(self, linear_wrapper):
        """
        Test the instantiate method when no default parameters are provided.
        """
        algorithm_instance = linear_wrapper.instantiate()
        assert isinstance(algorithm_instance, linear_wrapper.algorithm_class)
        assert algorithm_instance.wrapper_name == "linear"

    def test_instantiate_default_params(self, ridge_wrapper):
        """Test the instantiate method with default parameters."""
        algorithm_instance = ridge_wrapper.instantiate()
        assert isinstance(algorithm_instance, ridge_wrapper.algorithm_class)
        assert algorithm_instance.wrapper_name == "ridge"
        assert algorithm_instance.alpha == 1.0
        assert algorithm_instance.max_iter == 10000

    def test_instantiate_tuned_with_best_params(self, ridge_wrapper):
        """Test instantiate_tuned method with specific tuned parameters."""
        best_params = {"alpha": 1, "fit_intercept": False}
        algorithm_instance = ridge_wrapper.instantiate_tuned(best_params)
        assert algorithm_instance.wrapper_name == "ridge"
        assert algorithm_instance.alpha == 1
        assert algorithm_instance.fit_intercept is False

    def test_instantiate_tuned_with_defaults(self, rf_wrapper):
        """
        Test instantiate_tuned method ensures default parameters are applied
        """
        best_params = {
            "n_estimators": 40, 
            "criterion": "poisson",
        }
        algorithm_instance = rf_wrapper.instantiate_tuned(best_params)
        assert algorithm_instance.wrapper_name == "rf"
        assert algorithm_instance.n_estimators == 40
        assert algorithm_instance.criterion == "poisson"

    def test_instantiate_tuned_invalid(self, ridge_wrapper):
        """
        Check instantiate_tuned method raises TypeError for invalid parameters.
        """
        best_params = [100, True]
        with pytest.raises(TypeError, match="best_params must be a dictionary"):
            ridge_wrapper.instantiate_tuned(best_params)

    def test_to_markdown(self, ridge_wrapper, rf_wrapper):
        """Check the markdown representation of algorithm configurations."""
        expected_ridge_md = (
            "### Ridge Regression (`ridge`)\n"
            "\n"
            "- **Algorithm Class**: `Ridge`\n"
            "\n"
            "**Default Parameters:**\n"
            "```python\n"
            "'alpha': 1.0,\n"
            "'max_iter': 10000,\n"
            "```\n"
            "\n"
            "**Hyperparameter Grid:**\n"
            "```python\n"
            "'alpha': [0.1, 0.5, 1.0],\n"
            "```"
        )
        assert ridge_wrapper.to_markdown() == expected_ridge_md

        expected_rf_md = (
            "### Random Forest (`rf`)\n"
            "\n"
            "- **Algorithm Class**: `RandomForestRegressor`\n"
            "\n"
            "**Default Parameters:**\n"
            "```python\n"
            "'n_jobs': 1,\n"
            "```\n"
            "\n"
            "**Hyperparameter Grid:**\n"
            "```python\n"
            "'n_estimators': [20, 40, 60],\n"
            "```"
        )
        assert rf_wrapper.to_markdown() == expected_rf_md

    def test_export_config(self, ridge_wrapper):
        """Check the config dict has correct values"""
        config = ridge_wrapper.export_config()
        assert config["name"] == "ridge"
        assert config["display_name"] == "Ridge Regression"
        assert config["algorithm_class_module"] == "sklearn.linear_model._ridge"
        assert config["algorithm_class_name"] == "Ridge"
        assert config["default_params"] == {"alpha": 1.0, "max_iter": 10000}
        assert config["hyperparam_grid"] == {"alpha": [0.1, 0.5, 1.0]}
