"""Basic single-model workflow for regression e2e tests."""
from brisk.training.workflow import Workflow


class RegressionSingle(Workflow):
    """Single model regression workflow."""

    def workflow(self, X_train, X_test, y_train, y_test, output_dir, feature_names):
        metrics = ["MAE", "MSE", "R2"]
        model = self.model.fit(X_train, y_train)
        self.save_model(model, "fitted_model")
        self.evaluate_model(
            model, X_test, y_test, metrics, "eval_scores"
        )
        self.plot_residuals(
            model, X_test, y_test, "residual_plot"
        )
