"""Unit tests for AlgorithCollection"""
import pytest
from sklearn import linear_model, ensemble

from brisk import AlgorithmWrapper

from tests.utils.factories import AlgorithmFactory

# Prevent pytest fixtures from raising redefined-outer-name warning
# pylint: disable = redefined-outer-name

@pytest.fixture
def four_algorithm_collection():
    """Create an AlgorithCollection with 4 AlgorithmWrapper instances.
    """
    return AlgorithmFactory.collection(4, include_hyperparams=False)


class TestAlgorithmCollection():
    """All unit tests for AlgorithmCollection.
    
    Fixtures
    --------
    four_algorithm_collection: AlgorithmCollection
        Uses a factory to create an AlgorithmCollection instance
    """
    def test_str_index(self, four_algorithm_collection):
        """Access AlgorithmWrapper with a string
        """
        wrapper = four_algorithm_collection["linear"]
        assert wrapper.name == "linear"
        assert wrapper.display_name == "Linear Regression"
        assert wrapper.algorithm_class == linear_model.LinearRegression

        wrapper2 = four_algorithm_collection["ridge"]
        assert wrapper2.name == "ridge"
        assert wrapper2.display_name == "Ridge Regression"
        assert wrapper2.algorithm_class == linear_model.Ridge

    def test_missing_str_index(self, four_algorithm_collection):
        """Check error message when index does not exist"""
        with pytest.raises(
            KeyError, match="No algorithm found with name: "
        ):
            _ = four_algorithm_collection["not_a_wrapper"]

    def test_int_index(self, four_algorithm_collection):
        """Check int gives correct AlgorithmWrapper"""
        wrapper = four_algorithm_collection[1]
        assert wrapper.name == "ridge"
        assert wrapper.display_name == "Ridge Regression"
        assert wrapper.algorithm_class == linear_model.Ridge

        wrapper2 = four_algorithm_collection[3]
        assert wrapper2.name == "rf"
        assert wrapper2.display_name == "Random Forest"
        assert wrapper2.algorithm_class == ensemble.RandomForestRegressor

    def test_missing_int_index(self, four_algorithm_collection):
        """Check out of index int raises error"""
        with pytest.raises(
            IndexError, match="list index out of range"
        ):
            _ = four_algorithm_collection[5]

    def test_invalid_index_float(self, four_algorithm_collection):
        """Check using a float raises TypeError."""
        with pytest.raises(
            TypeError, match="Index must be an integer or string, got"
        ):
            _ =four_algorithm_collection[1.2]

    def test_invalid_index_list(self, four_algorithm_collection):
        """Check using a list raises TypeError."""
        with pytest.raises(
            TypeError, match="Index must be an integer or string, got"
        ):
            _ = four_algorithm_collection[["algo_name"]]

    def test_invalid_index_dict(self, four_algorithm_collection):
        """Check using a dict raises TypeError."""
        with pytest.raises(
            TypeError, match="Index must be an integer or string, got"
        ):
            _ = four_algorithm_collection[{"a dict": "is not allowed"}]

    def test_invalid_index_set(self, four_algorithm_collection):
        """Check using a set raises TypeError."""
        with pytest.raises(
            TypeError, match="Index must be an integer or string, got"
        ):
            _ = four_algorithm_collection[{"not", "a", "valid", "input"}]

    def test_invalid_append_int(self, four_algorithm_collection):
        """Check appending an int raises a TypeError."""
        with pytest.raises(
            TypeError,
            match="AlgorithmCollection only accepts AlgorithmWrapper instances"
        ):
            four_algorithm_collection.append(5)

    def test_invalid_append_string(self, four_algorithm_collection):
        """Check appending an str raises a TypeError."""
        with pytest.raises(
            TypeError,
            match="AlgorithmCollection only accepts AlgorithmWrapper instances"
        ):
            four_algorithm_collection.append("linear")

    def test_invalid_append_object(self, four_algorithm_collection):
        """
        Check appending an object that is not AlgorithmWrapper raises a
        TypeError.
        """
        with pytest.raises(
            TypeError,
            match="AlgorithmCollection only accepts AlgorithmWrapper instances"
        ):
            four_algorithm_collection.append(linear_model.LinearRegression)

    def test_duplicate_name_error(self, four_algorithm_collection):
        """Check the same name cannot be used for multiple AlgorithmWrappers"""
        with pytest.raises(
            ValueError,
            match="Duplicate algorithm name: linear"
        ):
            four_algorithm_collection.append(
                AlgorithmWrapper(
                    name="linear",
                    display_name="Linear Regression",
                    algorithm_class=linear_model.LinearRegression
                )
            )
