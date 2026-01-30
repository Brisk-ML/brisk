"""Algorithm configuration for regression e2e tests."""
import numpy as np
from sklearn import linear_model, svm, ensemble

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
        name="svc2",
        display_name="SVC (Second)",
        algorithm_class=svm.SVC
    ),
    brisk.AlgorithmWrapper(
        name="xtree",
        display_name="Extra Tree Regressor",
        algorithm_class=ensemble.ExtraTreesRegressor,
        default_params={"min_samples_split": 10},
        hyperparam_grid={
            "n_estimators": list(range(20, 160, 20)),
            "criterion": ["friedman_mse", "absolute_error", 
                          "poisson", "squared_error"],
            "max_depth": list(range(5, 25, 5)) + [None]
        }
    ),
    brisk.AlgorithmWrapper(
        name="linear_svc",
        display_name="Linear Support Vector Classification",
        algorithm_class=svm.LinearSVC,
        default_params={"max_iter": 10000},
        hyperparam_grid={
            "C": list(np.arange(1, 30, 0.5)), 
            "penalty": ["l1", "l2"],
        }
    )
)
