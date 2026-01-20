"""Unit tests for MetadataService."""
import json

import pytest
from sklearn import linear_model
from unittest import mock

from brisk.services import metadata
from tests.utils import factories

@pytest.fixture
def metadata_service():
    service = metadata.MetadataService("test_metadata")
    service.set_algorithm_config(factories.AlgorithmFactory.collection())
    return service


@pytest.fixture
def model():
    model = linear_model.Ridge()
    model.__setattr__("wrapper_name", "ridge")
    return model


@pytest.fixture
def two_models():
    model1 = linear_model.Ridge()
    model1.__setattr__("wrapper_name", "ridge")
    
    model2 = linear_model.LinearRegression()
    model2.__setattr__("wrapper_name", "linear")
    
    return [model1, model2]


class TestMetadataService:
    def test_get_model_has_base(self, metadata_service, model):
        """Test that get_model includes base metadata fields."""
        with mock.patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "2024-01-15 10:30:45"
            
            metadata = metadata_service.get_model(
                models=model,
                method_name="evaluate_model",
                is_test=False
            )
            assert metadata["timestamp"] == "2024-01-15 10:30:45"
            assert metadata["method"] == "evaluate_model"

    def test_get_model_json_serialize(self, metadata_service, model):
        """Test that get_model returns JSON-serializable metadata."""
        metadata = metadata_service.get_model(
            models=model,
            method_name="evaluate_model",
            is_test=False
        )
        
        json_str = json.dumps(metadata)
        assert isinstance(json_str, str)
        
        deserialized = json.loads(json_str)
        assert deserialized["type"] == "model"

    def test_get_model_one_model(self, metadata_service, model):
        """Test get_model with a single model."""
        metadata = metadata_service.get_model(
            models=model,
            method_name="evaluate_model",
            is_test=True
        )
        
        assert len(metadata["models"]) == 1
        assert metadata["models"]["ridge"] == "Ridge Regression"

    def test_get_model_two_models(self, metadata_service, two_models):
        """Test get_model with multiple models."""
        metadata = metadata_service.get_model(
            models=two_models,
            method_name="compare_models",
            is_test=False
        )
        
        assert len(metadata["models"]) == 2
        assert metadata["models"]["ridge"] == "Ridge Regression"
        assert metadata["models"]["linear"] == "Linear Regression"

    def test_get_dataset_has_base(self, metadata_service):
        """Test that get_dataset includes base metadata fields."""
        with mock.patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "2024-01-15 10:30:45"
            
            metadata = metadata_service.get_dataset(
                method_name="analyze_dataset",
                dataset_name="iris",
                group_name="classification"
            )
            
            assert metadata["timestamp"] == "2024-01-15 10:30:45"
            assert metadata["method"] == "analyze_dataset"

    def test_get_dataset_json_serialize(self, metadata_service):
        """Test that get_dataset returns JSON-serializable metadata."""
        metadata = metadata_service.get_dataset(
            method_name="analyze_dataset",
            dataset_name="iris",
            group_name="classification"
        )
        
        json_str = json.dumps(metadata)
        assert isinstance(json_str, str)
        
        deserialized = json.loads(json_str)
        assert deserialized["type"] == "dataset"

    def test_get_rerun_has_base(self, metadata_service):
        """Test that get_rerun includes base metadata fields."""
        with mock.patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "2024-01-15 10:30:45"
            
            metadata = metadata_service.get_rerun(
                method_name="save_rerun_config"
            )
            
            assert metadata["timestamp"] == "2024-01-15 10:30:45"
            assert metadata["method"] == "save_rerun_config"

    def test_get_rerun_json_serialize(self, metadata_service):
        """Test that get_rerun returns JSON-serializable metadata."""
        metadata = metadata_service.get_rerun(
            method_name="save_rerun_config"
        )
        
        json_str = json.dumps(metadata)
        assert isinstance(json_str, str)
        
        deserialized = json.loads(json_str)
        assert deserialized["type"] == "rerun_config"
