"""End-to-end tests for the brisk CLI run command.

This module tests the run command across:
- 3 problem types: regression, binary classification, multiclass classification
- 3 data formats: CSV, XLSX, SQLite DB
- 2 workflow types: single model, multi model

Total: 18 run test cases + 3 rerun test cases.
"""
import pytest

from tests.e2e.conftest import (
    E2ETestConfig,
    run_brisk_command,
    assert_run_success,
)


# =============================================================================
# Test Matrix Configuration
# =============================================================================

# All 18 test configurations: 3 problems × 3 formats × 2 workflows
RUN_TEST_CONFIGS = [
    # Regression tests
    pytest.param(
        E2ETestConfig(
            problem="regression",
            data_format="csv",
            workflow="single",
            dataset="continuous_features_regression.csv"
        ),
        id="regression_csv_single"
    ),
    pytest.param(
        E2ETestConfig(
            problem="regression",
            data_format="csv",
            workflow="multi",
            dataset="continuous_features_regression.csv"
        ),
        id="regression_csv_multi"
    ),
    pytest.param(
        E2ETestConfig(
            problem="regression",
            data_format="xlsx",
            workflow="single",
            dataset="categorical_features_regression.xlsx"
        ),
        id="regression_xlsx_single"
    ),
    pytest.param(
        E2ETestConfig(
            problem="regression",
            data_format="xlsx",
            workflow="multi",
            dataset="categorical_features_regression.xlsx"
        ),
        id="regression_xlsx_multi"
    ),
    pytest.param(
        E2ETestConfig(
            problem="regression",
            data_format="db",
            workflow="single",
            dataset=("mixed_features.db", "mixed_features_regression")
        ),
        id="regression_db_single"
    ),
    pytest.param(
        E2ETestConfig(
            problem="regression",
            data_format="db",
            workflow="multi",
            dataset=("mixed_features.db", "mixed_features_regression")
        ),
        id="regression_db_multi"
    ),

    # Binary classification tests
    pytest.param(
        E2ETestConfig(
            problem="binary",
            data_format="csv",
            workflow="single",
            dataset="continuous_features_binary.csv"
        ),
        id="binary_csv_single"
    ),
    pytest.param(
        E2ETestConfig(
            problem="binary",
            data_format="csv",
            workflow="multi",
            dataset="continuous_features_binary.csv"
        ),
        id="binary_csv_multi"
    ),
    pytest.param(
        E2ETestConfig(
            problem="binary",
            data_format="xlsx",
            workflow="single",
            dataset="categorical_features_binary.xlsx"
        ),
        id="binary_xlsx_single"
    ),
    pytest.param(
        E2ETestConfig(
            problem="binary",
            data_format="xlsx",
            workflow="multi",
            dataset="categorical_features_binary.xlsx"
        ),
        id="binary_xlsx_multi"
    ),
    pytest.param(
        E2ETestConfig(
            problem="binary",
            data_format="db",
            workflow="single",
            dataset=("mixed_features.db", "mixed_features_binary")
        ),
        id="binary_db_single"
    ),
    pytest.param(
        E2ETestConfig(
            problem="binary",
            data_format="db",
            workflow="multi",
            dataset=("mixed_features.db", "mixed_features_binary")
        ),
        id="binary_db_multi"
    ),

    # Multiclass classification tests
    pytest.param(
        E2ETestConfig(
            problem="multiclass",
            data_format="csv",
            workflow="single",
            dataset="continuous_features_categorical.csv"
        ),
        id="multiclass_csv_single"
    ),
    pytest.param(
        E2ETestConfig(
            problem="multiclass",
            data_format="csv",
            workflow="multi",
            dataset="continuous_features_categorical.csv"
        ),
        id="multiclass_csv_multi"
    ),
    pytest.param(
        E2ETestConfig(
            problem="multiclass",
            data_format="xlsx",
            workflow="single",
            dataset="categorical_features_categorical.xlsx"
        ),
        id="multiclass_xlsx_single"
    ),
    pytest.param(
        E2ETestConfig(
            problem="multiclass",
            data_format="xlsx",
            workflow="multi",
            dataset="categorical_features_categorical.xlsx"
        ),
        id="multiclass_xlsx_multi"
    ),
    pytest.param(
        E2ETestConfig(
            problem="multiclass",
            data_format="db",
            workflow="single",
            dataset=("mixed_features.db", "mixed_features_categorical")
        ),
        id="multiclass_db_single"
    ),
    pytest.param(
        E2ETestConfig(
            problem="multiclass",
            data_format="db",
            workflow="multi",
            dataset=("mixed_features.db", "mixed_features_categorical")
        ),
        id="multiclass_db_multi"
    ),
]


# =============================================================================
# Run Command Tests
# =============================================================================


@pytest.mark.e2e
class TestRunCommand:
    """Test the brisk run command across all configurations."""

    @pytest.mark.parametrize("e2e_project", RUN_TEST_CONFIGS, indirect=True)
    def test_run(self, e2e_project):
        """Test that brisk run completes successfully.

        Verifies:
        - Command exits with code 0
        - Results directory is created
        - Report HTML is generated
        - Run config JSON is saved (for rerun)
        - No experiments failed
        """
        config = e2e_project["config"]
        project_dir = e2e_project["project_dir"]

        result = run_brisk_command(
            project_dir=project_dir,
            results_name=config.test_id
        )

        assert_run_success(result)


# =============================================================================
# Rerun Command Tests
# =============================================================================

# Representative rerun test configurations (one per problem type,
# different formats)
RERUN_TEST_CONFIGS = [
    pytest.param(
        E2ETestConfig(
            problem="regression",
            data_format="db",
            workflow="single",
            dataset=("mixed_features.db", "mixed_features_regression")
        ),
        id="rerun_regression_db"
    ),
    pytest.param(
        E2ETestConfig(
            problem="binary",
            data_format="csv",
            workflow="single",
            dataset="continuous_features_binary.csv"
        ),
        id="rerun_binary_csv"
    ),
    pytest.param(
        E2ETestConfig(
            problem="multiclass",
            data_format="xlsx",
            workflow="single",
            dataset="categorical_features_categorical.xlsx"
        ),
        id="rerun_multiclass_xlsx"
    ),
]


@pytest.mark.e2e
class TestRerunCommand:
    """Test the brisk run -f (rerun) command functionality."""

    @pytest.mark.parametrize("e2e_project", RERUN_TEST_CONFIGS, indirect=True)
    def test_rerun(self, e2e_project):
        """Test that rerun from saved config works correctly.

        This test:
        1. Runs an initial experiment
        2. Reruns from the saved config
        3. Verifies both runs complete successfully
        """
        config = e2e_project["config"]
        project_dir = e2e_project["project_dir"]

        # Step 1: Initial run
        initial_run_name = f"{config.test_id}_initial"
        initial_result = run_brisk_command(
            project_dir=project_dir,
            results_name=initial_run_name
        )

        assert_run_success(initial_result)

        # Verify run_config.json exists (required for rerun)
        config_path = initial_result.results_dir / "run_config.json"
        assert config_path.exists(), (
            f"run_config.json not found at {config_path}"
        )

        # Step 2: Rerun from saved config
        rerun_name = f"{config.test_id}_rerun"
        rerun_result = run_brisk_command(
            project_dir=project_dir,
            results_name=rerun_name,
            config_file=initial_run_name
        )

        # Reruns don't create new run_config.json (they use coordinate mode)
        assert_run_success(rerun_result, is_rerun=True)
