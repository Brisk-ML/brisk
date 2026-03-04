"""Placeholder settings.py - will be overwritten by test fixtures."""
from brisk.configuration.configuration import Configuration


def create_configuration():
    """Placeholder configuration - overwritten by test fixtures."""
    config = Configuration(
        default_algorithms=["linear"]
    )
    config.add_experiment_group(
        name="placeholder",
        description="Placeholder group",
        datasets=[]
    )
    return config.build()
