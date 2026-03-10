"""Algorithm configuration for regression e2e tests."""
from sklearn import linear_model, tree

import brisk

ALGORITHM_CONFIG = brisk.AlgorithmCollection(
    *brisk.REGRESSION_ALGORITHMS,
    *brisk.CLASSIFICATION_ALGORITHMS,
    brisk.AlgorithmWrapper(
        name="linear2",
        display_name="Linear Regression (Second)",
        algorithm_class=linear_model.LinearRegression
    ),
    brisk.AlgorithmWrapper(
        name="dtc2",
        display_name="Decision Tree Classifier (Second)",
        algorithm_class=tree.DecisionTreeClassifier,
        default_params={"max_depth": 5},
        hyperparam_grid={"max_depth": [3, 5, 10]}
    ),
)
