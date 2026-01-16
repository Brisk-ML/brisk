from brisk.evaluation.evaluators import base

class UnregisteredOne(base.BaseEvaluator):
    def __init__(self, method_name, description):
        super().__init__(method_name, description)

class UnregisteredTwo(base.BaseEvaluator):
    def __init__(self, method_name, description):
        super().__init__(method_name, description)

def register_custom_evaluators(registry, theme):
    # Neither evaluator is registered
    pass
