"""Basic single-model workflow for binary classification e2e tests."""
from brisk.training.workflow import Workflow


class BinarySingle(Workflow):
    """Single model binary classification workflow."""

    def workflow(self, X_train, X_test, y_train, y_test, output_dir, feature_names):
        metrics = ["recall", "precision", "roc_auc"]
        model = self.model.fit(X_train, y_train)
        self.save_model(model, "fitted_model")
        self.plot_roc_curve(model, X_test, y_test, "roc_curve", 1)
        self.evaluate_model(
            model, X_test, y_test, metrics, "eval_model"
        )
        self.plot_precision_recall_curve(
            model, X_test, y_test, "precision_recall", 1
        )
