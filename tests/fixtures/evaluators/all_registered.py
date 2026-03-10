from brisk.evaluation.evaluators import base

class CustomEvaluator(base.BaseEvaluator):
    def __init__(self, method_name, description):
        super().__init__(method_name, description)
        self.name = "custom_evaluator"

def register_custom_evaluators(registry, theme):
    registry.register(CustomEvaluator(
        method_name="custom_evaluator",
        description="Test evaluator"
    ))
