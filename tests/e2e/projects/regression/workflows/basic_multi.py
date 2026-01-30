"""Basic multi-model workflow for regression e2e tests."""
from brisk.training.workflow import Workflow


class RegressionMulti(Workflow):
    """Multi-model regression workflow for comparing algorithms."""

    def workflow(self, X_train, X_test, y_train, y_test, output_dir, feature_names):
        metrics = ["MAE", "MSE", "R2"]
        model = self.model.fit(X_train, y_train)
        model2 = self.model2.fit(X_train, y_train)
        self.save_model(model, "fitted_model")
        self.save_model(model2, "fitted_model2")
        self.compare_models(
            model, model2, X=X_test, y=y_test, metrics=metrics,
            filename="compare"
        )
        self.plot_model_comparison(
            model, model2, X=X_test, y=y_test, metric="MAE",
            filename="plot_comparison"
        )
