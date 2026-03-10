"""Fixtures for E2E testing of the brisk CLI run command."""
import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import pytest

import numpy as np
from sklearn import metrics as sk_metrics

import brisk
from brisk.configuration import project

# pylint: disable=W0621, C0103

def huber_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Custom Huber loss metric for testing."""
    DELTA = 1
    loss = np.where(
        np.abs(y_true - y_pred) <= DELTA,
        0.5 * (y_true - y_pred)**2,
        DELTA * (np.abs(y_true - y_pred) - 0.5 * DELTA)
    )
    return np.mean(loss)


def fake_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_metadata: dict
) -> float:
    """Custom metric that uses split_metadata for testing."""
    return np.mean(
        (y_true - y_pred) /
        (split_metadata["num_features"] / split_metadata["num_samples"])
    )

def get_metric_config() -> brisk.MetricManager:
    """Create metric configuration with standard and custom metrics."""
    return brisk.MetricManager(
        *brisk.REGRESSION_METRICS,
        *brisk.CLASSIFICATION_METRICS,
        brisk.MetricWrapper(
            name="huber_loss",
            func=huber_loss,
            display_name="Huber Loss",
            greater_is_better=False
        ),
        brisk.MetricWrapper(
            name="fake_metric",
            func=fake_metric,
            display_name="Fake Metric",
            greater_is_better=False
        ),
        brisk.MetricWrapper(
            name="f1_multiclass",
            func=sk_metrics.f1_score,
            display_name="F1 Score (Multiclass)",
            greater_is_better=True,
            average="weighted"
        ),
        brisk.MetricWrapper(
            name="precision_multiclass",
            func=sk_metrics.precision_score,
            display_name="Precision (Multiclass)",
            greater_is_better=True,
            average="micro"
        ),
        brisk.MetricWrapper(
            name="recall_multiclass",
            func=sk_metrics.recall_score,
            display_name="Recall (Multiclass)",
            greater_is_better=True,
            average="macro"
        ),
    )



# =============================================================================
# Dataset Configuration
# =============================================================================

# Categorical feature mappings for datasets that require them
CATEGORICAL_FEATURES: Dict[str, List[str]] = {
    "categorical_features_regression.xlsx": [
        "categorical_0", "categorical_1", "categorical_2",
        "categorical_3", "categorical_4", "categorical_5",
        "categorical_6", "categorical_7", "categorical_8",
        "categorical_9"
    ],
    "categorical_features_binary.xlsx": [
        "categorical_0", "categorical_1", "categorical_2",
        "categorical_3", "categorical_4", "categorical_5",
        "categorical_6", "categorical_7"
    ],
    "categorical_features_categorical.xlsx": [
        "categorical_0", "categorical_1", "categorical_2",
        "categorical_3", "categorical_4", "categorical_5",
        "categorical_6"
    ],
    ("mixed_features.db", "mixed_features_regression"): [
        "categorical_0", "categorical_1", "categorical_2"
    ],
    ("mixed_features.db", "mixed_features_binary"): [
        "categorical_0", "categorical_1", "categorical_2",
        "categorical_3"
    ],
    ("mixed_features.db", "mixed_features_categorical"): [
        "categorical_0", "categorical_1", "categorical_2"
    ],
}

# Algorithm selections by problem type
ALGORITHMS_BY_PROBLEM = {
    "regression": {
        "single": ["linear", "lasso", "ridge"],
        "multi": [["linear", "lasso"], ["ridge", "elasticnet"]],
    },
    "binary": {
        "single": ["logistic", "dtc", "ridge_classifier"],
        "multi": [["logistic", "dtc"], ["gaussian_nb", "ridge_classifier"]],
    },
    "multiclass": {
        "single": ["logistic", "dtc", "gaussian_nb"],
        "multi": [["logistic", "dtc"], ["gaussian_nb", "ridge_classifier"]],
    },
}


# =============================================================================
# Data Classes for Test Configuration
# =============================================================================

@dataclass
class E2ETestConfig:
    """Configuration for a single e2e test case."""
    problem: str  # regression, binary, multiclass
    data_format: str  # csv, xlsx, db
    workflow: str  # single, multi
    dataset: Union[str, Tuple[str, str]]  # filename or (filename, table_name)

    @property
    def test_id(self) -> str:
        """Generate unique test identifier."""
        return f"{self.problem}_{self.data_format}_{self.workflow}"

    @property
    def project_type(self) -> str:
        """Map problem type to project directory name."""
        if self.problem == "multiclass":
            return "classification"
        return self.problem

    @property
    def algorithms(self) -> Union[List[str], List[List[str]]]:
        """Get algorithms for this test configuration."""
        return ALGORITHMS_BY_PROBLEM[self.problem][self.workflow]

    @property
    def categorical_features(self) -> Optional[Dict]:
        """Get categorical feature configuration if needed."""
        if self.data_format == "csv":
            return None

        key = self.dataset if isinstance(self.dataset, tuple) else self.dataset
        if key in CATEGORICAL_FEATURES:
            return {self.dataset: CATEGORICAL_FEATURES[key]}
        return None


@dataclass
class RunResult:
    """Result of running a brisk command."""
    returncode: int
    stdout: str
    stderr: str
    results_dir: pathlib.Path


# =============================================================================
# Helper Functions
# =============================================================================

def run_brisk_command(
    project_dir: pathlib.Path,
    results_name: str,
    config_file: Optional[str] = None
) -> RunResult:
    """Execute the brisk run command.

    Parameters
    ----------
    project_dir : Path
        Path to the project directory.
    results_name : str
        Name for the results directory.
    config_file : str, optional
        Name of config file for rerun (uses -f flag).

    Returns
    -------
    RunResult
        Contains returncode, stdout, stderr, and results_dir path.

    Notes
    -----
    The workflow is specified in the project's settings.py file,
    not as a CLI argument.
    """
    project_root = str(pathlib.Path(__file__).parent.parent.parent)
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    env = os.environ.copy()
    if current_pythonpath:
        env["PYTHONPATH"] = f"{project_root}:{current_pythonpath}"
    else:
        env["PYTHONPATH"] = project_root

    cmd = ["brisk", "run", "-n", results_name]

    if config_file:
        cmd.extend(["-f", config_file])

    use_shell = sys.platform == "win32"

    result = subprocess.run(
        cmd,
        cwd=str(project_dir),
        capture_output=True,
        text=True,
        check=False,
        env=env,
        shell=use_shell
    )

    results_dir = project_dir / "results" / results_name

    return RunResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        results_dir=results_dir
    )


def assert_run_success(result: RunResult, is_rerun: bool = False) -> None:
    """Assert that a brisk run completed successfully.

    Parameters
    ----------
    result : RunResult
        The result from run_brisk_command.
    is_rerun : bool, default=False
        If True, skip the run_config.json check (reruns don't create
        new configs).

    Raises
    ------
    AssertionError
        If any success criteria are not met.
    """
    assert result.returncode == 0, (
        f"Command failed with return code {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )

    assert result.results_dir.exists(), (
        f"Results directory not created: {result.results_dir}"
    )

    assert (result.results_dir / "report.html").exists(), (
        f"Report not generated in {result.results_dir}"
    )

    # run_config.json is only created in capture mode, not in coordinate
    # (rerun) mode
    if not is_rerun:
        assert (result.results_dir / "run_config.json").exists(), (
            f"Rerun config not saved in {result.results_dir}"
        )

    assert "FAILED" not in result.stdout, (
        f"Experiment failures detected in output:\n{result.stdout}"
    )


def generate_settings_content(config: E2ETestConfig, workflow_name: str) -> str:
    """Generate settings.py content for a test configuration.

    Parameters
    ----------
    config : E2ETestConfig
        The test configuration.
    workflow_name : str
        Name of the workflow file (without .py extension).

    Returns
    -------
    str
        Content for the settings.py file.
    """
    # Build categorical features dict string
    cat_features_str = "None"
    if config.categorical_features:
        items = []
        for key, features in config.categorical_features.items():
            if isinstance(key, tuple):
                key_str = f'("{key[0]}", "{key[1]}")'
            else:
                key_str = f'"{key}"'
            features_str = repr(features)
            items.append(f"        {key_str}: {features_str}")
        cat_features_str = "{\n" + ",\n".join(items) + "\n    }"

    # Build algorithms string
    algorithms = config.algorithms
    algo_str = repr(algorithms)

    # Build dataset string
    if isinstance(config.dataset, tuple):
        dataset_str = f'[("{config.dataset[0]}", "{config.dataset[1]}")]'
    else:
        dataset_str = f'["{config.dataset}"]'

    # Determine preprocessors based on data format
    preprocessors = []
    if config.data_format in ("xlsx", "db"):
        preprocessors.append('CategoricalEncodingPreprocessor(method="onehot")')
    if config.data_format == "csv":
        preprocessors.append('ScalingPreprocessor(method="standard")')

    preprocessors_str = ", ".join(preprocessors) if preprocessors else ""
    data_config_str = ""
    if preprocessors_str:
        data_config_str = f"""
            data_config={{
                "preprocessors": [{preprocessors_str}]
            }},"""

    return f'''"""Auto-generated settings for e2e test: {config.test_id}"""
from brisk.configuration.configuration import Configuration
from brisk.data.preprocessing import (
    ScalingPreprocessor,
    CategoricalEncodingPreprocessor,
    FeatureSelectionPreprocessor,
    MissingDataPreprocessor
)


def create_configuration():
    config = Configuration(
        default_workflow="{workflow_name}",
        default_algorithms={algo_str},
        categorical_features={cat_features_str}
    )
    config.add_experiment_group(
        name="test_group",
        description="E2E test experiment group",
        datasets={dataset_str},{data_config_str}
    )
    return config.build()
'''


def setup_project(
    tmp_path: pathlib.Path,
    config: E2ETestConfig,
    workflow_name: str,
    projects_dir: pathlib.Path,
    datasets_dir: pathlib.Path
) -> pathlib.Path:
    """Set up a test project directory.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for the test.
    config : E2ETestConfig
        Test configuration.
    workflow_name : str
        Name of the workflow file (without .py extension).
    projects_dir : Path
        Path to the e2e/projects directory with templates.
    datasets_dir : Path
        Path to the fixtures/datasets directory.

    Returns
    -------
    Path
        Path to the set up project directory.
    """
    # Copy project template
    project_template = projects_dir / config.project_type
    project_dir = tmp_path / config.project_type
    shutil.copytree(project_template, project_dir)

    # Create datasets directory and copy required datasets
    project_datasets = project_dir / "datasets"
    project_datasets.mkdir(exist_ok=True)

    if isinstance(config.dataset, tuple):
        # Database file - copy the db file
        src = datasets_dir / config.dataset[0]
        dst = project_datasets / config.dataset[0]
        shutil.copy2(src, dst)
    else:
        # Single file - copy it
        src = datasets_dir / config.dataset
        dst = project_datasets / config.dataset
        shutil.copy2(src, dst)

    # Generate and write settings.py
    settings_content = generate_settings_content(config, workflow_name)
    settings_path = project_dir / "settings.py"
    settings_path.write_text(settings_content)

    # Clear any existing results
    results_dir = project_dir / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)

    return project_dir


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def datasets_dir() -> pathlib.Path:
    """Path to the fixtures/datasets directory."""
    return pathlib.Path(__file__).parent.parent / "fixtures" / "datasets"


@pytest.fixture(scope="session")
def projects_dir() -> pathlib.Path:
    """Path to the e2e/projects directory with project templates."""
    return pathlib.Path(__file__).parent / "projects"


@pytest.fixture
def e2e_project(request, tmp_path, projects_dir, datasets_dir):
    """
    Fixture that sets up a project for e2e testing.

    Use with indirect parametrization to pass E2ETestConfig.
    """
    config: E2ETestConfig = request.param
    workflow_name = f"basic_{config.workflow}"

    # Clear project root cache
    project.find_project_root.cache_clear()

    project_dir = setup_project(
        tmp_path, config, workflow_name, projects_dir, datasets_dir
    )

    yield {
        "project_dir": project_dir,
        "config": config,
    }
