"""Integration tests for CLI commands.

This module tests the CLI commands including:
- create: Project initialization
- export-env: Environment export to requirements.txt
- check-env: Environment compatibility checking

The environment commands are tested together to ensure they properly
detect matching and differing environments.
"""
import json
import os
import pathlib
import platform
import subprocess
import sys
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from brisk.cli.cli import cli, create, export_env, check_env
from brisk.cli.environment import EnvironmentManager
from brisk.configuration import project


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def clear_project_root_cache():
    """Clear the project root cache before each test."""
    project.find_project_root.cache_clear()


@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def project_dir(tmp_path, monkeypatch):
    """Create a temporary project directory with briskconfig."""
    proj_dir = tmp_path / "test_project"
    proj_dir.mkdir()
    (proj_dir / ".briskconfig").write_text("project_name=test_project\n")
    monkeypatch.chdir(proj_dir)
    return proj_dir


@pytest.fixture
def mock_packages():
    """Mock package list for environment tests."""
    return [
        {"name": "numpy", "version": "1.24.0"},
        {"name": "pandas", "version": "2.0.0"},
        {"name": "scikit-learn", "version": "1.3.0"},
        {"name": "scipy", "version": "1.11.0"},
        {"name": "joblib", "version": "1.3.0"},
        {"name": "requests", "version": "2.28.0"},
    ]


def create_run_config(
    results_dir: pathlib.Path,
    env_data: dict
) -> pathlib.Path:
    """Create a mock run_config.json file with environment data.
    
    Parameters
    ----------
    results_dir : pathlib.Path
        Directory to create the config in
    env_data : dict
        Environment data to include
        
    Returns
    -------
    pathlib.Path
        Path to the created config file
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    config_path = results_dir / "run_config.json"
    
    config = {
        "env": env_data,
        "experiment_groups": []
    }
    
    config_path.write_text(json.dumps(config, indent=2))
    return config_path


# =============================================================================
# Create Command Tests
# =============================================================================

class TestCreateCommand:
    """Tests for the 'brisk create' command."""

    def test_create_project_directory(self, cli_runner, tmp_path, monkeypatch):
        """Test that create command creates project directory."""
        monkeypatch.chdir(tmp_path)
        
        result = cli_runner.invoke(create, ["-n", "my_project"])
        
        assert result.exit_code == 0
        assert (tmp_path / "my_project").exists()
        assert "new project was created" in result.output

    def test_create_briskconfig(self, cli_runner, tmp_path, monkeypatch):
        """Test that create command creates .briskconfig file."""
        monkeypatch.chdir(tmp_path)
        
        result = cli_runner.invoke(create, ["-n", "my_project"])
        
        assert result.exit_code == 0
        config_file = tmp_path / "my_project" / ".briskconfig"
        assert config_file.exists()
        content = config_file.read_text()
        assert "project_name=my_project" in content

    def test_create_settings_py(self, cli_runner, tmp_path, monkeypatch):
        """Test that create command creates settings.py file."""
        monkeypatch.chdir(tmp_path)
        
        result = cli_runner.invoke(create, ["-n", "my_project"])
        
        assert result.exit_code == 0
        settings_file = tmp_path / "my_project" / "settings.py"
        assert settings_file.exists()
        content = settings_file.read_text()
        assert "from brisk.configuration.configuration import Configuration" in content
        assert "def create_configuration()" in content
        assert "ConfigurationManager" in content

    def test_create_algorithms_py(self, cli_runner, tmp_path, monkeypatch):
        """Test that create command creates algorithms.py file."""
        monkeypatch.chdir(tmp_path)
        
        result = cli_runner.invoke(create, ["-n", "my_project"])
        
        assert result.exit_code == 0
        algorithms_file = tmp_path / "my_project" / "algorithms.py"
        assert algorithms_file.exists()
        content = algorithms_file.read_text()
        assert "AlgorithmCollection" in content
        assert "REGRESSION_ALGORITHMS" in content
        assert "CLASSIFICATION_ALGORITHMS" in content

    def test_create_metrics_py(self, cli_runner, tmp_path, monkeypatch):
        """Test that create command creates metrics.py file."""
        monkeypatch.chdir(tmp_path)
        
        result = cli_runner.invoke(create, ["-n", "my_project"])
        
        assert result.exit_code == 0
        metrics_file = tmp_path / "my_project" / "metrics.py"
        assert metrics_file.exists()
        content = metrics_file.read_text()
        assert "MetricManager" in content
        assert "REGRESSION_METRICS" in content

    def test_create_data_py(self, cli_runner, tmp_path, monkeypatch):
        """Test that create command creates data.py file."""
        monkeypatch.chdir(tmp_path)
        
        result = cli_runner.invoke(create, ["-n", "my_project"])
        
        assert result.exit_code == 0
        data_file = tmp_path / "my_project" / "data.py"
        assert data_file.exists()
        content = data_file.read_text()
        assert "DataManager" in content
        assert "test_size" in content

    def test_create_evaluators_py(self, cli_runner, tmp_path, monkeypatch):
        """Test that create command creates evaluators.py file."""
        monkeypatch.chdir(tmp_path)
        
        result = cli_runner.invoke(create, ["-n", "my_project"])
        
        assert result.exit_code == 0
        evaluators_file = tmp_path / "my_project" / "evaluators.py"
        assert evaluators_file.exists()
        content = evaluators_file.read_text()
        assert "EvaluatorRegistry" in content
        assert "register_custom_evaluators" in content

    def test_create_workflows_directory(self, cli_runner, tmp_path, monkeypatch):
        """Test that create command creates workflows directory."""
        monkeypatch.chdir(tmp_path)
        
        result = cli_runner.invoke(create, ["-n", "my_project"])
        
        assert result.exit_code == 0
        workflows_dir = tmp_path / "my_project" / "workflows"
        assert workflows_dir.exists()
        assert workflows_dir.is_dir()

    def test_create_workflow_template(self, cli_runner, tmp_path, monkeypatch):
        """Test that create command creates workflow.py template."""
        monkeypatch.chdir(tmp_path)
        
        result = cli_runner.invoke(create, ["-n", "my_project"])
        
        assert result.exit_code == 0
        workflow_file = tmp_path / "my_project" / "workflows" / "workflow.py"
        assert workflow_file.exists()
        content = workflow_file.read_text()
        assert "class MyWorkflow(Workflow)" in content
        assert "def workflow(" in content

    def test_create_datasets_directory(self, cli_runner, tmp_path, monkeypatch):
        """Test that create command creates datasets directory."""
        monkeypatch.chdir(tmp_path)
        
        result = cli_runner.invoke(create, ["-n", "my_project"])
        
        assert result.exit_code == 0
        datasets_dir = tmp_path / "my_project" / "datasets"
        assert datasets_dir.exists()
        assert datasets_dir.is_dir()

    def test_create_all_expected_files(self, cli_runner, tmp_path, monkeypatch):
        """Test that create command creates all expected files."""
        monkeypatch.chdir(tmp_path)
        
        result = cli_runner.invoke(create, ["-n", "my_project"])
        
        assert result.exit_code == 0
        project_dir = tmp_path / "my_project"
        
        expected_files = [
            ".briskconfig",
            "settings.py",
            "algorithms.py",
            "metrics.py",
            "data.py",
            "evaluators.py",
            "workflows/workflow.py",
        ]
        
        expected_dirs = [
            "datasets",
            "workflows",
        ]
        
        for file_path in expected_files:
            assert (project_dir / file_path).exists(), f"Missing: {file_path}"
        
        for dir_path in expected_dirs:
            assert (project_dir / dir_path).is_dir(), f"Missing dir: {dir_path}"


# =============================================================================
# Export-Env Command Tests
# =============================================================================

class TestExportEnvCommand:
    """Tests for the 'brisk export-env' command."""

    def test_export_env_creates_requirements_file(
        self, cli_runner, project_dir
    ):
        """Test that export-env creates a requirements.txt file."""
        run_id = "test_run_001"
        results_dir = project_dir / "results" / run_id
        
        env_data = {
            "timestamp": "2024-01-01T12:00:00",
            "python": {"version": "3.10.0"},
            "packages": {
                "numpy": {"version": "1.24.0", "is_critical": True},
                "pandas": {"version": "2.0.0", "is_critical": True},
            }
        }
        create_run_config(results_dir, env_data)
        
        result = cli_runner.invoke(export_env, [run_id])
        
        assert result.exit_code == 0
        assert "Requirements exported to" in result.output

    def test_export_env_custom_output_path(self, cli_runner, project_dir):
        """Test export-env with custom output path."""
        run_id = "test_run_002"
        results_dir = project_dir / "results" / run_id
        
        env_data = {
            "timestamp": "2024-01-01T12:00:00",
            "python": {"version": "3.10.0"},
            "packages": {
                "numpy": {"version": "1.24.0", "is_critical": True},
            }
        }
        create_run_config(results_dir, env_data)
        
        output_path = project_dir / "custom_requirements.txt"
        result = cli_runner.invoke(
            export_env, [run_id, "--output", str(output_path)]
        )
        
        assert result.exit_code == 0
        assert output_path.exists()
        content = output_path.read_text()
        assert "numpy==1.24.0" in content

    def test_export_env_include_all(self, cli_runner, project_dir):
        """Test export-env with --include-all flag."""
        run_id = "test_run_003"
        results_dir = project_dir / "results" / run_id
        
        env_data = {
            "timestamp": "2024-01-01T12:00:00",
            "python": {"version": "3.10.0"},
            "packages": {
                "numpy": {"version": "1.24.0", "is_critical": True},
                "requests": {"version": "2.28.0", "is_critical": False},
            }
        }
        create_run_config(results_dir, env_data)
        
        output_path = project_dir / "all_requirements.txt"
        result = cli_runner.invoke(
            export_env, [run_id, "--output", str(output_path), "--include-all"]
        )
        
        assert result.exit_code == 0
        content = output_path.read_text()
        assert "numpy==1.24.0" in content
        assert "requests==2.28.0" in content

    def test_export_env_critical_only(self, cli_runner, project_dir):
        """Test export-env exports only critical packages by default."""
        run_id = "test_run_004"
        results_dir = project_dir / "results" / run_id
        
        env_data = {
            "timestamp": "2024-01-01T12:00:00",
            "python": {"version": "3.10.0"},
            "packages": {
                "numpy": {"version": "1.24.0", "is_critical": True},
                "requests": {"version": "2.28.0", "is_critical": False},
            }
        }
        create_run_config(results_dir, env_data)
        
        output_path = project_dir / "critical_requirements.txt"
        result = cli_runner.invoke(
            export_env, [run_id, "--output", str(output_path)]
        )
        
        assert result.exit_code == 0
        content = output_path.read_text()
        assert "numpy==1.24.0" in content
        assert "requests" not in content

    def test_export_env_run_not_found(self, cli_runner, project_dir):
        """Test export-env with non-existent run ID."""
        result = cli_runner.invoke(export_env, ["nonexistent_run"])
        
        assert "Error: Run configuration not found" in result.output

    def test_export_env_no_environment_data(self, cli_runner, project_dir):
        """Test export-env when config has no environment data."""
        run_id = "test_run_005"
        results_dir = project_dir / "results" / run_id
        results_dir.mkdir(parents=True)
        
        config_path = results_dir / "run_config.json"
        config_path.write_text(json.dumps({"experiment_groups": []}))
        
        result = cli_runner.invoke(export_env, [run_id])
        
        assert "No environment information found" in result.output


# =============================================================================
# Check-Env Command Tests
# =============================================================================

class TestCheckEnvCommand:
    """Tests for the 'brisk check-env' command."""

    @patch.object(EnvironmentManager, "_get_current_packages_dict")
    @patch("brisk.cli.environment.platform.python_version")
    def test_check_env_compatible(
        self, mock_python_version, mock_packages,
        cli_runner, project_dir
    ):
        """Test check-env with compatible environment."""
        mock_python_version.return_value = "3.10.0"
        mock_packages.return_value = {
            "numpy": "1.24.0",
            "pandas": "2.0.0",
        }
        
        run_id = "test_run_006"
        results_dir = project_dir / "results" / run_id
        
        env_data = {
            "python": {"version": "3.10.0"},
            "packages": {
                "numpy": {"version": "1.24.0", "is_critical": True},
                "pandas": {"version": "2.0.0", "is_critical": True},
            }
        }
        create_run_config(results_dir, env_data)
        
        result = cli_runner.invoke(check_env, [run_id])
        
        assert result.exit_code == 0
        assert "compatible" in result.output.lower()

    @patch.object(EnvironmentManager, "_get_current_packages_dict")
    @patch("brisk.cli.environment.platform.python_version")
    def test_check_env_incompatible_critical_package(
        self, mock_python_version, mock_packages,
        cli_runner, project_dir
    ):
        """Test check-env detects incompatible critical package."""
        mock_python_version.return_value = "3.10.0"
        mock_packages.return_value = {
            "numpy": "2.0.0",  # Major version change
            "pandas": "2.0.0",
        }
        
        run_id = "test_run_007"
        results_dir = project_dir / "results" / run_id
        
        env_data = {
            "python": {"version": "3.10.0"},
            "packages": {
                "numpy": {"version": "1.24.0", "is_critical": True},
                "pandas": {"version": "2.0.0", "is_critical": True},
            }
        }
        create_run_config(results_dir, env_data)
        
        result = cli_runner.invoke(check_env, [run_id])
        
        assert result.exit_code == 0
        assert "critical differences" in result.output.lower()

    @patch.object(EnvironmentManager, "_get_current_packages_dict")
    @patch("brisk.cli.environment.platform.python_version")
    def test_check_env_missing_package(
        self, mock_python_version, mock_packages,
        cli_runner, project_dir
    ):
        """Test check-env detects missing package."""
        mock_python_version.return_value = "3.10.0"
        mock_packages.return_value = {
            "pandas": "2.0.0",
            # numpy is missing
        }
        
        run_id = "test_run_008"
        results_dir = project_dir / "results" / run_id
        
        env_data = {
            "python": {"version": "3.10.0"},
            "packages": {
                "numpy": {"version": "1.24.0", "is_critical": True},
                "pandas": {"version": "2.0.0", "is_critical": True},
            }
        }
        create_run_config(results_dir, env_data)
        
        result = cli_runner.invoke(check_env, [run_id])
        
        assert result.exit_code == 0
        assert "critical differences" in result.output.lower()

    @patch.object(EnvironmentManager, "_get_current_packages_dict")
    @patch("brisk.cli.environment.platform.python_version")
    def test_check_env_verbose(
        self, mock_python_version, mock_packages,
        cli_runner, project_dir
    ):
        """Test check-env with verbose flag shows detailed report."""
        mock_python_version.return_value = "3.10.0"
        mock_packages.return_value = {
            "numpy": "1.24.0",
        }
        
        run_id = "test_run_009"
        results_dir = project_dir / "results" / run_id
        
        env_data = {
            "python": {"version": "3.10.0"},
            "system": {"platform": "Linux-5.15.0"},
            "packages": {
                "numpy": {"version": "1.24.0", "is_critical": True},
            }
        }
        create_run_config(results_dir, env_data)
        
        result = cli_runner.invoke(check_env, [run_id, "--verbose"])
        
        assert result.exit_code == 0
        assert "ENVIRONMENT COMPATIBILITY REPORT" in result.output
        assert "Python Version" in result.output

    def test_check_env_run_not_found(self, cli_runner, project_dir):
        """Test check-env with non-existent run ID."""
        result = cli_runner.invoke(check_env, ["nonexistent_run"])
        
        assert "Error: Run configuration not found" in result.output

    def test_check_env_no_environment_data(self, cli_runner, project_dir):
        """Test check-env when config has no environment data."""
        run_id = "test_run_010"
        results_dir = project_dir / "results" / run_id
        results_dir.mkdir(parents=True)
        
        config_path = results_dir / "run_config.json"
        config_path.write_text(json.dumps({"experiment_groups": []}))
        
        result = cli_runner.invoke(check_env, [run_id])
        
        assert "No environment information found" in result.output


# =============================================================================
# Environment Commands Integration Tests
# =============================================================================

class TestEnvironmentCommandsIntegration:
    """Integration tests ensuring export-env and check-env work together.
    
    These tests verify that:
    1. When environments match, check-env passes
    2. When environments differ, check-env detects the differences
    3. export-env produces a valid requirements file that can recreate the env
    """

    @patch.object(EnvironmentManager, "_get_current_packages_dict")
    @patch("brisk.cli.environment.platform.python_version")
    def test_matching_environments_pass(
        self, mock_python_version, mock_packages,
        cli_runner, project_dir
    ):
        """Test that identical environments pass check-env."""
        # Current environment matches saved
        mock_python_version.return_value = "3.10.0"
        mock_packages.return_value = {
            "numpy": "1.24.0",
            "pandas": "2.0.0",
            "scikit-learn": "1.3.0",
        }
        
        run_id = "matching_env_test"
        results_dir = project_dir / "results" / run_id
        
        # Save environment that matches current
        env_data = {
            "timestamp": "2024-01-01T12:00:00",
            "python": {"version": "3.10.0"},
            "packages": {
                "numpy": {"version": "1.24.0", "is_critical": True},
                "pandas": {"version": "2.0.0", "is_critical": True},
                "scikit-learn": {"version": "1.3.0", "is_critical": True},
            }
        }
        create_run_config(results_dir, env_data)
        
        # Check-env should report compatible
        result = cli_runner.invoke(check_env, [run_id])
        
        assert result.exit_code == 0
        assert "compatible" in result.output.lower()
        assert "critical differences" not in result.output.lower()

    @patch.object(EnvironmentManager, "_get_current_packages_dict")
    @patch("brisk.cli.environment.platform.python_version")
    def test_different_environments_detected(
        self, mock_python_version, mock_packages,
        cli_runner, project_dir
    ):
        """Test that different environments are detected by check-env."""
        # Current environment differs from saved
        mock_python_version.return_value = "3.10.0"
        mock_packages.return_value = {
            "numpy": "2.0.0",  # Different major version
            "pandas": "2.1.0",  # Different minor version (critical)
            "scikit-learn": "1.3.0",
        }
        
        run_id = "different_env_test"
        results_dir = project_dir / "results" / run_id
        
        env_data = {
            "timestamp": "2024-01-01T12:00:00",
            "python": {"version": "3.10.0"},
            "packages": {
                "numpy": {"version": "1.24.0", "is_critical": True},
                "pandas": {"version": "2.0.0", "is_critical": True},
                "scikit-learn": {"version": "1.3.0", "is_critical": True},
            }
        }
        create_run_config(results_dir, env_data)
        
        # Check-env should report differences
        result = cli_runner.invoke(check_env, [run_id])
        
        assert result.exit_code == 0
        assert "critical differences" in result.output.lower()

    @patch.object(EnvironmentManager, "_get_current_packages_dict")
    @patch("brisk.cli.environment.platform.python_version")
    def test_python_version_mismatch_detected(
        self, mock_python_version, mock_packages,
        cli_runner, project_dir
    ):
        """Test that Python version mismatch is detected."""
        mock_python_version.return_value = "3.11.0"  # Different Python
        mock_packages.return_value = {
            "numpy": "1.24.0",
        }
        
        run_id = "python_mismatch_test"
        results_dir = project_dir / "results" / run_id
        
        env_data = {
            "timestamp": "2024-01-01T12:00:00",
            "python": {"version": "3.10.0"},  # Saved as 3.10
            "packages": {
                "numpy": {"version": "1.24.0", "is_critical": True},
            }
        }
        create_run_config(results_dir, env_data)
        
        result = cli_runner.invoke(check_env, [run_id])
        
        assert result.exit_code == 0
        assert "critical differences" in result.output.lower()

    @patch.object(EnvironmentManager, "_get_current_packages_dict")
    @patch("brisk.cli.environment.platform.python_version")
    def test_export_then_check_workflow(
        self, mock_python_version, mock_packages,
        cli_runner, project_dir
    ):
        """Test full workflow: export-env then check-env on same config."""
        mock_python_version.return_value = "3.10.0"
        mock_packages.return_value = {
            "numpy": "1.24.0",
            "pandas": "2.0.0",
        }
        
        run_id = "workflow_test"
        results_dir = project_dir / "results" / run_id
        
        env_data = {
            "timestamp": "2024-01-01T12:00:00",
            "python": {"version": "3.10.0"},
            "packages": {
                "numpy": {"version": "1.24.0", "is_critical": True},
                "pandas": {"version": "2.0.0", "is_critical": True},
            }
        }
        create_run_config(results_dir, env_data)
        
        # Step 1: Export requirements
        output_path = project_dir / "requirements.txt"
        export_result = cli_runner.invoke(
            export_env, [run_id, "--output", str(output_path)]
        )
        
        assert export_result.exit_code == 0
        assert output_path.exists()
        
        # Verify requirements content
        requirements_content = output_path.read_text()
        assert "numpy==1.24.0" in requirements_content
        assert "pandas==2.0.0" in requirements_content
        
        # Step 2: Check environment - should be compatible
        check_result = cli_runner.invoke(check_env, [run_id])
        
        assert check_result.exit_code == 0
        assert "compatible" in check_result.output.lower()

    @patch.object(EnvironmentManager, "_get_current_packages_dict")
    @patch("brisk.cli.environment.platform.python_version")
    def test_patch_version_changes_compatible(
        self, mock_python_version, mock_packages,
        cli_runner, project_dir
    ):
        """Test that patch version changes are considered compatible."""
        mock_python_version.return_value = "3.10.5"  # Patch difference
        mock_packages.return_value = {
            "numpy": "1.24.3",  # Patch difference
            "pandas": "2.0.1",  # Patch difference
        }
        
        run_id = "patch_version_test"
        results_dir = project_dir / "results" / run_id
        
        env_data = {
            "timestamp": "2024-01-01T12:00:00",
            "python": {"version": "3.10.0"},
            "packages": {
                "numpy": {"version": "1.24.0", "is_critical": True},
                "pandas": {"version": "2.0.0", "is_critical": True},
            }
        }
        create_run_config(results_dir, env_data)
        
        result = cli_runner.invoke(check_env, [run_id])
        
        assert result.exit_code == 0
        # Patch differences should be compatible
        assert "compatible" in result.output.lower()

    @patch.object(EnvironmentManager, "_get_current_packages_dict")
    @patch("brisk.cli.environment.platform.python_version")
    def test_extra_packages_compatible(
        self, mock_python_version, mock_packages,
        cli_runner, project_dir
    ):
        """Test that extra packages don't break compatibility."""
        mock_python_version.return_value = "3.10.0"
        mock_packages.return_value = {
            "numpy": "1.24.0",
            "pandas": "2.0.0",
            "matplotlib": "3.8.0",  # Extra package
        }
        
        run_id = "extra_packages_test"
        results_dir = project_dir / "results" / run_id
        
        env_data = {
            "timestamp": "2024-01-01T12:00:00",
            "python": {"version": "3.10.0"},
            "packages": {
                "numpy": {"version": "1.24.0", "is_critical": True},
                "pandas": {"version": "2.0.0", "is_critical": True},
            }
        }
        create_run_config(results_dir, env_data)
        
        result = cli_runner.invoke(check_env, [run_id])
        
        assert result.exit_code == 0
        # Extra packages should still be compatible
        assert "compatible" in result.output.lower()

    @patch.object(EnvironmentManager, "_get_current_packages_dict")
    @patch("brisk.cli.environment.platform.python_version")
    def test_missing_non_critical_package_incompatible(
        self, mock_python_version, mock_packages,
        cli_runner, project_dir
    ):
        """Test that missing non-critical packages break compatibility."""
        mock_python_version.return_value = "3.10.0"
        mock_packages.return_value = {
            "numpy": "1.24.0",
            # requests is missing
        }
        
        run_id = "missing_non_critical_test"
        results_dir = project_dir / "results" / run_id
        
        env_data = {
            "timestamp": "2024-01-01T12:00:00",
            "python": {"version": "3.10.0"},
            "packages": {
                "numpy": {"version": "1.24.0", "is_critical": True},
                "requests": {"version": "2.28.0", "is_critical": False},
            }
        }
        create_run_config(results_dir, env_data)
        
        result = cli_runner.invoke(check_env, [run_id])
        
        assert result.exit_code == 0
        # Missing packages should break compatibility
        assert "critical differences" in result.output.lower()

    @patch.object(EnvironmentManager, "_get_current_packages_dict")
    @patch("brisk.cli.environment.platform.python_version")
    @patch("brisk.cli.environment.platform.platform")
    def test_verbose_shows_detailed_report(
        self, mock_platform, mock_python_version, mock_packages,
        cli_runner, project_dir
    ):
        """Test that verbose flag shows comprehensive report."""
        mock_python_version.return_value = "3.10.0"
        mock_platform.return_value = "Linux-5.15.0-generic"
        mock_packages.return_value = {
            "numpy": "2.0.0",  # Version changed
            "pandas": "2.0.0",
        }
        
        run_id = "verbose_report_test"
        results_dir = project_dir / "results" / run_id
        
        env_data = {
            "timestamp": "2024-01-01T12:00:00",
            "python": {"version": "3.10.0"},
            "system": {"platform": "Linux-5.15.0-generic"},
            "packages": {
                "numpy": {"version": "1.24.0", "is_critical": True},
                "pandas": {"version": "2.0.0", "is_critical": True},
            }
        }
        create_run_config(results_dir, env_data)
        
        result = cli_runner.invoke(check_env, [run_id, "--verbose"])
        
        assert result.exit_code == 0
        output = result.output
        
        # Verify report structure
        assert "ENVIRONMENT COMPATIBILITY REPORT" in output
        assert "Python Version" in output
        assert "3.10.0" in output
        
        # Should show the numpy version difference
        assert "numpy" in output.lower()
        assert "RECOMMENDATION" in output
