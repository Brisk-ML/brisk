"""Basic single-model workflow for multiclass classification e2e tests."""
from brisk.training.workflow import Workflow


class ClassificationSingle(Workflow):
    """Single model multiclass classification workflow."""

    def workflow(self, X_train, X_test, y_train, y_test, output_dir, feature_names):
        metrics = ["accuracy", "balanced_accuracy"]
        model = self.model.fit(X_train, y_train)
        self.save_model(model, "fitted_model")
        self.plot_confusion_heatmap(
            model, X_test, y_test, "heatmap"
        )
        self.confusion_matrix(
            model, X_test, y_test, "confusion_matrix"
        )
        self.evaluate_model(
            model, X_test, y_test, metrics, "eval_model"
        )
