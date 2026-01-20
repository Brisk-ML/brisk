"""Unit tests for Workflow."""

import pytest
import pandas as pd
import numpy as np
from unittest import mock
from sklearn import ensemble, linear_model

from brisk.training import workflow


class ExampleWorkflow(workflow.Workflow):
    """Concrete implementation of Workflow for testing."""
    
    def workflow(self, X_train, X_test, y_train, y_test, output_dir, feature_names):
        """Simple workflow implementation for testing."""
        pass


@pytest.fixture
def mock_evaluation_manager():
    """Create a mock EvaluationManager."""
    manager = mock.MagicMock()
    manager.get_evaluator = mock.MagicMock(return_value=mock.MagicMock())
    manager.save_model = mock.MagicMock()
    manager.load_model = mock.MagicMock()
    return manager


@pytest.fixture
def sample_data():
    """Create sample training and test data."""
    np.random.seed(42)
    
    X_train = pd.DataFrame(
        np.random.randn(100, 3),
        columns=['feature_0', 'feature_1', 'feature_2']
    )
    X_test = pd.DataFrame(
        np.random.randn(30, 3),
        columns=['feature_0', 'feature_1', 'feature_2']
    )
    y_train = pd.Series(np.random.randn(100), name='target')
    y_test = pd.Series(np.random.randn(30), name='target')
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


@pytest.fixture
def workflow_params(mock_evaluation_manager, sample_data):
    """Create standard workflow parameters."""
    return {
        'evaluation_manager': mock_evaluation_manager,
        'X_train': sample_data['X_train'],
        'X_test': sample_data['X_test'],
        'y_train': sample_data['y_train'],
        'y_test': sample_data['y_test'],
        'output_dir': '/path/to/output',
        'algorithm_names': ['RandomForest', 'LinearRegression'],
        'feature_names': ['feature_0', 'feature_1', 'feature_2'],
        'workflow_attributes': {}
    }


class TestWorkflow:
    def test_unpack_attributes_one_attribute(self, workflow_params):
        """Test workflow construction with one additional attribute."""
        model = ensemble.RandomForestClassifier()
        workflow_params['workflow_attributes'] = {'model': model}
        
        workflow = ExampleWorkflow(**workflow_params)
        
        assert workflow.model is model

    def test_unpack_attributes_two_attributes(self, workflow_params):
        """Test workflow construction with two additional attributes."""
        model1 = ensemble.RandomForestClassifier()
        model2 = linear_model.LinearRegression()
        workflow_params['workflow_attributes'] = {
            'model1': model1,
            'model2': model2
        }
        
        workflow = ExampleWorkflow(**workflow_params)
        
        assert workflow.model1 is model1
        assert workflow.model2 is model2

    def test_data_attrs_preserved(self, workflow_params):
        """Test that is_test attributes are properly set on data."""
        workflow = ExampleWorkflow(**workflow_params)
        
        assert workflow.X_train.attrs['is_test'] is False
        assert workflow.y_train.attrs['is_test'] is False
        
        assert workflow.X_test.attrs['is_test'] is True
        assert workflow.y_test.attrs['is_test'] is True

    def test_workflow_not_implemented(self):
        """Test that Workflow cannot be instantiated without implementing workflow()."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            workflow.Workflow(
                evaluation_manager=mock.MagicMock(),
                X_train=pd.DataFrame(),
                X_test=pd.DataFrame(),
                y_train=pd.Series(dtype=float),
                y_test=pd.Series(dtype=float),
                output_dir='/path',
                algorithm_names=[],
                feature_names=[],
                workflow_attributes={}
            )
