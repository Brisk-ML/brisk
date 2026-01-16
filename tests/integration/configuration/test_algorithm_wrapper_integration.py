"""Integration tests for AlgorithmWrapper."""
import json

from sklearn import linear_model

from brisk.configuration import algorithm_wrapper

class TestAlgorithmWrapperIntegration():
    def test_export_config_serializable(self):
        wrapper = algorithm_wrapper.AlgorithmWrapper(
            name="test_wrapper",
            display_name="Test Wrapper",
            algorithm_class = linear_model.Ridge
        )
        output = wrapper.export_config()

        output_json = json.dumps(output, indent=2)
        deserialized = json.loads(output_json)

        assert deserialized == output

    def test_export_config_algorithm_module(self):
        wrapper = algorithm_wrapper.AlgorithmWrapper(
            name="test_wrapper",
            display_name="Test Wrapper",
            algorithm_class = linear_model.Ridge
        )
        output = wrapper.export_config()

        assert output["algorithm_class_module"] == "sklearn.linear_model._ridge"
        assert output["algorithm_class_name"] == "Ridge"
