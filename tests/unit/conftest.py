"""Fixtures for use in unit tests."""

import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_services():
    """Mock service layer for testing."""
    services = MagicMock()
    return services
