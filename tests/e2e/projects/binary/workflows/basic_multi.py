"""Basic multi-model workflow for binary classification e2e tests."""
from brisk.training.workflow import Workflow


class BinaryMulti(Workflow):
    """Multi-model binary classification workflow for comparing algorithms."""

    def workflow(self, X_train, X_test, y_train, y_test, output_dir, feature_names):
        metrics = ["accuracy", "f1", "precision", "recall"]
        model = self.model.fit(X_train, y_train)
        model2 = self.model2.fit(X_train, y_train)
        self.save_model(model, "fitted_model")
        self.save_model(model2, "fitted_model2")
        self.compare_models(
            model, model2, X=X_test, y=y_test, metrics=metrics, 
            filename="compare"
        )
        self.plot_model_comparison(
            model, model2, X=X_test, y=y_test, metric="f1",
            filename="plot_comparison"
        )
