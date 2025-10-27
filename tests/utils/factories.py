"""Factories to create objects for testing."""
from typing import Dict, Any

from sklearn import linear_model

from brisk.configuration.algorithm_wrapper import AlgorithmWrapper

class AlgorithmFactory():
    """Factory to create AlgorithmWrapper instances for use in tests."""
    @classmethod
    def simple(cls):
        return AlgorithmWrapper(
            name="ridge",
            display_name="Ridge Regression",
            algorithm_class=linear_model.Ridge,
            default_params={"alpha": 0.5},
            hyperparam_grid={"alpha": [0.1, 0.5, 1.0]}
        )

    @classmethod
    def full(
        cls,
        name: str = "ridge",
        display_name: str = "Ridge Regression",
        algorithm_class=linear_model.Ridge,
        default_params: Dict[str, Any] = {"alpha": 0.5},
        hyperparam_grid: Dict[str, Any] = {"alpha": [0.1, 0.5, 1.0]}
    ):
        return AlgorithmWrapper(
            name=name,
            display_name=display_name,
            algorithm_class=algorithm_class,
            default_params=default_params,
            hyperparam_grid=hyperparam_grid
        )
