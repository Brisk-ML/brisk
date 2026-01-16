"""Integration tests for ExperimentGroup."""
import pytest

from brisk.configuration import project
from brisk.configuration.experiment_group import ExperimentGroup

class TestExperimentGroupIntegration:
    def test_missing_dataset_error(self, tmp_path):
        with project.ProjectRootContext(tmp_path):
            with pytest.raises(FileNotFoundError):
                group = ExperimentGroup(
                    name="test",
                    datasets=["fake.csv"],
                    workflow="workflow",
                    algorithms=["linear"]
                )
