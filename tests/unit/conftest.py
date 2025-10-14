# Pytest fixtures for unit/ tests
from pathlib import Path

import pytest

from brisk import services as services_module
from brisk.configuration import project as project_module
from brisk.configuration import configuration as configuration_module
from brisk.configuration import configuration_manager as config_manager_module

from tests.utils.mocks import MockServiceBundle

# All unit tests should be isolated from the services layer. Any tests to
# check service integration should be written in integration/ as setting up
# service layer is expensive and makes unit testing run too slowly.
@pytest.fixture(autouse=True)
def mock_get_services(monkeypatch):
    """Global patch for brisk.services.get_services used by all unit tests."""
    mock_bundle = MockServiceBundle()

    def mock_get_services_func():
        return mock_bundle


    monkeypatch.setattr(services_module, "get_services", mock_get_services_func)
    monkeypatch.setattr(
        config_manager_module, "get_services", mock_get_services_func
    )
    monkeypatch.setattr(
        configuration_module, "get_services", mock_get_services_func
    )
    yield


# Unit tests should not be using real file operations. These should be tested in
# the integration tests. find_project_root() is mocked for unit tests so file
# paths cannot be resolved properly by Brisk.
@pytest.fixture(autouse=True)
def mock_get_project_root(monkeypatch):
    """Global patch for brisk.configuration.project.find_project_root()."""
    def mock_find_project_root_func():
        return Path("./") 


    monkeypatch.setattr(
        project_module, "find_project_root", mock_find_project_root_func
    )
    yield
