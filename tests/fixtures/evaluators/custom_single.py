from brisk.evaluation.evaluators import base

class CustomEvaluator(base.BaseEvaluator):
    """A custom evaluator for testing."""

    def __init__(self, method_name, description):
        super().__init__(method_name, description)
        self.name = "custom_evaluator"

    def evaluate(self, model, X_test, y_test):
        return {"custom_metric": 0.95}

def register_custom_evaluators(registry, theme):
    registry.register(CustomEvaluator(
        method_name="custom_evaluator",
        description="For testing"
    ))
