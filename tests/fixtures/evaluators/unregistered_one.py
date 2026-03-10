from brisk.evaluation.evaluators import base

class RegisteredEvaluator(base.BaseEvaluator):
    def __init__(self, method_name, description):
        super().__init__(method_name, description)
        self.name = "registered"

class UnregisteredEvaluator(base.BaseEvaluator):
    def __init__(self, method_name, description):
        super().__init__(method_name, description)
        self.name = "unregistered"

def register_custom_evaluators(registry, theme):
    registry.register(RegisteredEvaluator(
        method_name="registered",
        description="Registered evaluator"
    ))
    # UnregisteredEvaluator is intentionally not registered
