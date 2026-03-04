"""Unit tests for the environment module.

This module tests the EnvironmentManager class and related data structures
for capturing, comparing, and exporting Python environments.
"""
from unittest.mock import patch, MagicMock

import pytest

from brisk.cli.environment import (
    VersionMatch,
    EnvironmentDiff,
    EnvironmentManager,
)

# pylint: disable=W0212

@pytest.mark.unit
class TestVersionMatch:
    """Tests for the VersionMatch enumeration."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert VersionMatch.EXACT.value == "exact"
        assert VersionMatch.COMPATIBLE.value == "compatible"
        assert VersionMatch.INCOMPATIBLE.value == "incompatible"
        assert VersionMatch.MISSING.value == "missing"
        assert VersionMatch.EXTRA.value == "extra"

    def test_enum_membership(self):
        """Test enum membership checks."""
        assert VersionMatch.EXACT in VersionMatch
        assert VersionMatch.COMPATIBLE in VersionMatch


# =============================================================================
# EnvironmentDiff Tests
# =============================================================================


@pytest.mark.unit
class TestEnvironmentDiff:
    """Tests for the EnvironmentDiff dataclass."""

    def test_str_missing_package(self):
        """Test string representation for missing packages."""
        diff = EnvironmentDiff(
            package="numpy",
            original_version="1.24.0",
            current_version=None,
            status=VersionMatch.MISSING,
            is_critical=True
        )
        result = str(diff)
        assert "numpy==1.24.0" in result
        assert "not installed" in result
        assert "[CRITICAL]" in result

    def test_str_missing_non_critical(self):
        """Test string representation for missing non-critical packages."""
        diff = EnvironmentDiff(
            package="requests",
            original_version="2.28.0",
            current_version=None,
            status=VersionMatch.MISSING,
            is_critical=False
        )
        result = str(diff)
        assert "requests==2.28.0" in result
        assert "not installed" in result
        assert "[CRITICAL]" not in result

    def test_str_extra_package(self):
        """Test string representation for extra packages."""
        diff = EnvironmentDiff(
            package="matplotlib",
            original_version=None,
            current_version="3.8.0",
            status=VersionMatch.EXTRA,
            is_critical=False
        )
        result = str(diff)
        assert "matplotlib==3.8.0" in result
        assert "not in original" in result

    def test_str_incompatible_critical(self):
        """Test string representation for incompatible critical packages."""
        diff = EnvironmentDiff(
            package="pandas",
            original_version="1.5.0",
            current_version="2.0.0",
            status=VersionMatch.INCOMPATIBLE,
            is_critical=True
        )
        result = str(diff)
        assert "pandas" in result
        assert "1.5.0" in result
        assert "2.0.0" in result
        assert "major/minor version change" in result
        assert "[CRITICAL]" in result

    def test_str_incompatible_non_critical(self):
        """Test string representation for incompatible non-critical packages."""
        diff = EnvironmentDiff(
            package="requests",
            original_version="2.0.0",
            current_version="3.0.0",
            status=VersionMatch.INCOMPATIBLE,
            is_critical=False
        )
        result = str(diff)
        assert "requests" in result
        assert "2.0.0" in result
        assert "3.0.0" in result
        assert "major version change" in result
        assert "[CRITICAL]" not in result

    def test_str_compatible(self):
        """Test string representation for compatible version changes."""
        diff = EnvironmentDiff(
            package="numpy",
            original_version="1.24.0",
            current_version="1.24.1",
            status=VersionMatch.COMPATIBLE,
            is_critical=True
        )
        result = str(diff)
        assert "numpy" in result
        assert "1.24.0" in result
        assert "1.24.1" in result
        assert "patch version change" in result

    def test_str_exact(self):
        """Test string representation for exact matches."""
        diff = EnvironmentDiff(
            package="scipy",
            original_version="1.11.0",
            current_version="1.11.0",
            status=VersionMatch.EXACT,
            is_critical=True
        )
        result = str(diff)
        assert "scipy==1.11.0" in result


# =============================================================================
# EnvironmentManager Tests
# =============================================================================


@pytest.mark.unit
class TestEnvironmentManager:
    """Tests for the EnvironmentManager class."""

    @pytest.fixture
    def env_manager(self, tmp_path):
        """Create an EnvironmentManager for testing."""
        return EnvironmentManager(tmp_path)

    # -------------------------------------------------------------------------
    # Version Comparison Tests
    # -------------------------------------------------------------------------

    class TestCompareVersions:
        """Tests for _compare_versions method."""

        @pytest.fixture
        def env_manager(self, tmp_path):
            return EnvironmentManager(tmp_path)

        def test_exact_match(self, env_manager):
            """Test that identical versions return EXACT."""
            result = env_manager._compare_versions("1.2.3", "1.2.3")
            assert result == VersionMatch.EXACT

        def test_major_version_change(self, env_manager):
            """Test that major version changes return INCOMPATIBLE."""
            result = env_manager._compare_versions("1.2.3", "2.0.0")
            assert result == VersionMatch.INCOMPATIBLE

        def test_minor_version_change_non_critical(self, env_manager):
            """Test minor version change for non-critical package."""
            result = env_manager._compare_versions(
                "1.2.3", "1.5.0", is_critical=False
            )
            assert result == VersionMatch.COMPATIBLE

        def test_minor_version_change_critical(self, env_manager):
            """Test minor version change for critical package."""
            result = env_manager._compare_versions(
                "1.2.3", "1.5.0", is_critical=True
            )
            assert result == VersionMatch.INCOMPATIBLE

        def test_patch_version_change_critical(self, env_manager):
            """Test patch version change for critical package."""
            result = env_manager._compare_versions(
                "1.2.3", "1.2.5", is_critical=True
            )
            assert result == VersionMatch.COMPATIBLE

        def test_patch_version_change_non_critical(self, env_manager):
            """Test patch version change for non-critical package."""
            result = env_manager._compare_versions(
                "1.2.3", "1.2.5", is_critical=False
            )
            assert result == VersionMatch.COMPATIBLE

    # -------------------------------------------------------------------------
    # Environment Capture Tests
    # -------------------------------------------------------------------------

    class TestCaptureEnvironment:
        """Tests for capture_environment method."""

        @pytest.fixture
        def env_manager(self, tmp_path):
            return EnvironmentManager(tmp_path)

        @patch("brisk.cli.environment.subprocess.run")
        def test_capture_environment_structure(
            self, mock_run, env_manager
        ):
            """Test that captured environment has correct structure."""
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='[{"name": "numpy", "version": "1.24.0"}]'
            )

            env = env_manager.capture_environment()

            assert "python" in env
            assert "version" in env["python"]
            assert "implementation" in env["python"]
            assert "system" in env
            assert "platform" in env["system"]
            assert "architecture" in env["system"]
            assert "packages" in env

        @patch("brisk.cli.environment.subprocess.run")
        def test_capture_packages_marks_critical(
            self, mock_run, env_manager
        ):
            """Test that critical packages are marked correctly."""
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='[{"name": "numpy", "version": "1.24.0"}, '
                       '{"name": "requests", "version": "2.28.0"}]'
            )

            env = env_manager.capture_environment()

            assert env["packages"]["numpy"]["is_critical"] is True
            assert env["packages"]["requests"]["is_critical"] is False

        @patch("brisk.cli.environment.subprocess.run")
        def test_capture_handles_json_decode_error(
            self, mock_run, env_manager, capsys
        ):
            """Test that JSON decode errors are handled gracefully."""
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="invalid json"
            )

            env = env_manager.capture_environment()

            assert env["packages"] == {}
            captured = capsys.readouterr()
            assert "Warning" in captured.out

    # -------------------------------------------------------------------------
    # Environment Comparison Tests
    # -------------------------------------------------------------------------

    class TestCompareEnvironments:
        """Tests for compare_environments method."""

        @pytest.fixture
        def env_manager(self, tmp_path):
            return EnvironmentManager(tmp_path)

        @patch.object(EnvironmentManager, "_get_current_packages_dict")
        @patch("brisk.cli.environment.platform.python_version")
        def test_identical_environments(
            self, mock_python_version, mock_packages, env_manager
        ):
            """Test comparing identical environments."""
            mock_python_version.return_value = "3.10.0"
            mock_packages.return_value = {
                "numpy": "1.24.0",
                "pandas": "2.0.0",
            }

            saved_env = {
                "python": {"version": "3.10.0"},
                "packages": {
                    "numpy": {"version": "1.24.0", "is_critical": True},
                    "pandas": {"version": "2.0.0", "is_critical": True},
                }
            }

            differences, is_compatible = env_manager.compare_environments(
                saved_env
            )

            assert len(differences) == 0
            assert is_compatible is True

        @patch.object(EnvironmentManager, "_get_current_packages_dict")
        @patch("brisk.cli.environment.platform.python_version")
        def test_missing_critical_package(
            self, mock_python_version, mock_packages, env_manager
        ):
            """Test that missing critical packages break compatibility."""
            mock_python_version.return_value = "3.10.0"
            mock_packages.return_value = {
                "pandas": "2.0.0",
            }

            saved_env = {
                "python": {"version": "3.10.0"},
                "packages": {
                    "numpy": {"version": "1.24.0", "is_critical": True},
                    "pandas": {"version": "2.0.0", "is_critical": True},
                }
            }

            differences, is_compatible = env_manager.compare_environments(
                saved_env
            )

            assert is_compatible is False
            missing_numpy = [
                d for d in differences
                if d.package == "numpy" and d.status == VersionMatch.MISSING
            ]
            assert len(missing_numpy) == 1

        @patch.object(EnvironmentManager, "_get_current_packages_dict")
        @patch("brisk.cli.environment.platform.python_version")
        def test_extra_package(
            self, mock_python_version, mock_packages, env_manager
        ):
            """Test that extra packages are detected."""
            mock_python_version.return_value = "3.10.0"
            mock_packages.return_value = {
                "numpy": "1.24.0",
                "matplotlib": "3.8.0",
            }

            saved_env = {
                "python": {"version": "3.10.0"},
                "packages": {
                    "numpy": {"version": "1.24.0", "is_critical": True},
                }
            }

            differences, is_compatible = env_manager.compare_environments(
                saved_env
            )

            extra_matplotlib = [
                d for d in differences
                if d.package == "matplotlib" and d.status == VersionMatch.EXTRA
            ]
            assert len(extra_matplotlib) == 1
            # Extra packages don't break compatibility
            assert is_compatible is True

        @patch.object(EnvironmentManager, "_get_current_packages_dict")
        @patch("brisk.cli.environment.platform.python_version")
        def test_python_version_mismatch(
            self, mock_python_version, mock_packages, env_manager
        ):
            """Test that Python major.minor version mismatch breaks compat."""
            mock_python_version.return_value = "3.11.0"
            mock_packages.return_value = {"numpy": "1.24.0"}

            saved_env = {
                "python": {"version": "3.10.0"},
                "packages": {
                    "numpy": {"version": "1.24.0", "is_critical": True},
                }
            }

            differences, is_compatible = env_manager.compare_environments(
                saved_env
            )

            assert is_compatible is False
            python_diff = [
                d for d in differences
                if d.package == "python"
            ]
            assert len(python_diff) == 1
            assert python_diff[0].status == VersionMatch.INCOMPATIBLE

        @patch.object(EnvironmentManager, "_get_current_packages_dict")
        @patch("brisk.cli.environment.platform.python_version")
        def test_python_patch_version_compatible(
            self, mock_python_version, mock_packages, env_manager
        ):
            """Test that Python patch version differences are allowed."""
            mock_python_version.return_value = "3.10.5"
            mock_packages.return_value = {"numpy": "1.24.0"}

            saved_env = {
                "python": {"version": "3.10.0"},
                "packages": {
                    "numpy": {"version": "1.24.0", "is_critical": True},
                }
            }

            differences, is_compatible = env_manager.compare_environments(
                saved_env
            )

            # Patch version change doesn't add to differences for Python
            python_diff = [d for d in differences if d.package == "python"]
            assert len(python_diff) == 0
            assert is_compatible is True

        @patch.object(EnvironmentManager, "_get_current_packages_dict")
        @patch("brisk.cli.environment.platform.python_version")
        def test_critical_incompatible_version(
            self, mock_python_version, mock_packages, env_manager
        ):
            """Test critical package with incompatible version."""
            mock_python_version.return_value = "3.10.0"
            mock_packages.return_value = {"numpy": "2.0.0"}

            saved_env = {
                "python": {"version": "3.10.0"},
                "packages": {
                    "numpy": {"version": "1.24.0", "is_critical": True},
                }
            }

            differences, is_compatible = env_manager.compare_environments(
                saved_env
            )

            assert is_compatible is False
            numpy_diff = [d for d in differences if d.package == "numpy"]
            assert len(numpy_diff) == 1
            assert numpy_diff[0].status == VersionMatch.INCOMPATIBLE

    # -------------------------------------------------------------------------
    # Requirements Export Tests
    # -------------------------------------------------------------------------

    class TestExportRequirements:
        """Tests for export_requirements method."""

        @pytest.fixture
        def env_manager(self, tmp_path):
            return EnvironmentManager(tmp_path)

        def test_export_critical_only(self, env_manager, tmp_path):
            """Test exporting only critical packages."""
            saved_env = {
                "timestamp": "2024-01-01T12:00:00",
                "python": {"version": "3.10.0"},
                "packages": {
                    "numpy": {"version": "1.24.0", "is_critical": True},
                    "pandas": {"version": "2.0.0", "is_critical": True},
                    "requests": {"version": "2.28.0", "is_critical": False},
                }
            }

            output_path = tmp_path / "requirements.txt"
            result = env_manager.export_requirements(
                saved_env, output_path, include_all=False
            )

            content = result.read_text()
            assert "numpy==1.24.0" in content
            assert "pandas==2.0.0" in content
            assert "requests==2.28.0" not in content
            assert "# Critical packages" in content

        def test_export_all_packages(self, env_manager, tmp_path):
            """Test exporting all packages."""
            saved_env = {
                "timestamp": "2024-01-01T12:00:00",
                "python": {"version": "3.10.0"},
                "packages": {
                    "numpy": {"version": "1.24.0", "is_critical": True},
                    "requests": {"version": "2.28.0", "is_critical": False},
                }
            }

            output_path = tmp_path / "requirements.txt"
            result = env_manager.export_requirements(
                saved_env, output_path, include_all=True
            )

            content = result.read_text()
            assert "numpy==1.24.0" in content
            assert "requests==2.28.0" in content
            assert "# Critical packages" in content
            assert "# Other packages" in content

        def test_export_includes_python_version(self, env_manager, tmp_path):
            """Test that Python version is included in header."""
            saved_env = {
                "timestamp": "2024-01-01T12:00:00",
                "python": {"version": "3.10.0"},
                "packages": {
                    "numpy": {"version": "1.24.0", "is_critical": True},
                }
            }

            output_path = tmp_path / "requirements.txt"
            result = env_manager.export_requirements(
                saved_env, output_path, include_python=True
            )

            content = result.read_text()
            assert "# Python version: 3.10.0" in content

        def test_export_without_python_version(self, env_manager, tmp_path):
            """Test excluding Python version from header."""
            saved_env = {
                "timestamp": "2024-01-01T12:00:00",
                "python": {"version": "3.10.0"},
                "packages": {
                    "numpy": {"version": "1.24.0", "is_critical": True},
                }
            }

            output_path = tmp_path / "requirements.txt"
            result = env_manager.export_requirements(
                saved_env, output_path, include_python=False
            )

            content = result.read_text()
            assert "Python version" not in content

        def test_export_creates_parent_directories(self, env_manager, tmp_path):
            """Test that parent directories are created if needed."""
            saved_env = {
                "timestamp": "2024-01-01T12:00:00",
                "python": {"version": "3.10.0"},
                "packages": {
                    "numpy": {"version": "1.24.0", "is_critical": True},
                }
            }

            output_path = tmp_path / "nested" / "dir" / "requirements.txt"
            result = env_manager.export_requirements(saved_env, output_path)

            assert result.exists()
            assert result.parent.exists()

    # -------------------------------------------------------------------------
    # Categorize Differences Tests
    # -------------------------------------------------------------------------

    class TestCategorizeDifferences:
        """Tests for _categorize_differences method."""

        @pytest.fixture
        def env_manager(self, tmp_path):
            return EnvironmentManager(tmp_path)

        def test_categorizes_critical_missing(self, env_manager):
            """Test categorization of missing critical packages."""
            differences = [
                EnvironmentDiff(
                    package="numpy",
                    original_version="1.24.0",
                    current_version=None,
                    status=VersionMatch.MISSING,
                    is_critical=True
                )
            ]

            categories = env_manager._categorize_differences(differences)

            assert len(categories["critical_missing"]) == 1
            assert categories["critical_missing"][0].package == "numpy"

        def test_categorizes_extra_packages(self, env_manager):
            """Test categorization of extra packages."""
            differences = [
                EnvironmentDiff(
                    package="matplotlib",
                    original_version=None,
                    current_version="3.8.0",
                    status=VersionMatch.EXTRA,
                    is_critical=False
                )
            ]

            categories = env_manager._categorize_differences(differences)

            assert len(categories["extra"]) == 1
            assert categories["extra"][0].package == "matplotlib"

        def test_categorizes_all_types(self, env_manager):
            """Test categorization of multiple difference types."""
            differences = [
                EnvironmentDiff(
                    package="numpy", original_version="1.24.0",
                    current_version=None, status=VersionMatch.MISSING,
                    is_critical=True
                ),
                EnvironmentDiff(
                    package="pandas", original_version="1.5.0",
                    current_version="2.0.0", status=VersionMatch.INCOMPATIBLE,
                    is_critical=True
                ),
                EnvironmentDiff(
                    package="scipy", original_version="1.10.0",
                    current_version="1.10.1", status=VersionMatch.COMPATIBLE,
                    is_critical=True
                ),
                EnvironmentDiff(
                    package="requests", original_version="2.28.0",
                    current_version=None, status=VersionMatch.MISSING,
                    is_critical=False
                ),
                EnvironmentDiff(
                    package="flask", original_version="2.0.0",
                    current_version="3.0.0", status=VersionMatch.INCOMPATIBLE,
                    is_critical=False
                ),
                EnvironmentDiff(
                    package="django", original_version="4.0.0",
                    current_version="4.0.1", status=VersionMatch.COMPATIBLE,
                    is_critical=False
                ),
                EnvironmentDiff(
                    package="matplotlib", original_version=None,
                    current_version="3.8.0", status=VersionMatch.EXTRA,
                    is_critical=False
                ),
            ]

            categories = env_manager._categorize_differences(differences)

            assert len(categories["critical_missing"]) == 1
            assert len(categories["critical_incompatible"]) == 1
            assert len(categories["critical_compatible"]) == 1
            assert len(categories["non_critical_missing"]) == 1
            assert len(categories["non_critical_incompatible"]) == 1
            assert len(categories["non_critical_compatible"]) == 1
            assert len(categories["extra"]) == 1

    # -------------------------------------------------------------------------
    # Report Generation Tests
    # -------------------------------------------------------------------------

    class TestGenerateEnvironmentReport:
        """Tests for generate_environment_report method."""

        @pytest.fixture
        def env_manager(self, tmp_path):
            return EnvironmentManager(tmp_path)

        @patch.object(EnvironmentManager, "_get_current_packages_dict")
        @patch("brisk.cli.environment.platform.python_version")
        @patch("brisk.cli.environment.platform.platform")
        def test_report_compatible_environment(
            self, mock_platform, mock_python_version,
            mock_packages, env_manager
        ):
            """Test report for compatible environment."""
            mock_python_version.return_value = "3.10.0"
            mock_platform.return_value = "Linux-5.15.0"
            mock_packages.return_value = {"numpy": "1.24.0"}

            saved_env = {
                "python": {"version": "3.10.0"},
                "system": {"platform": "Linux-5.15.0"},
                "packages": {
                    "numpy": {"version": "1.24.0", "is_critical": True},
                }
            }

            report = env_manager.generate_environment_report(saved_env)

            assert "ENVIRONMENT COMPATIBILITY REPORT" in report
            assert "compatible" in report.lower()

        @patch.object(EnvironmentManager, "_get_current_packages_dict")
        @patch("brisk.cli.environment.platform.python_version")
        @patch("brisk.cli.environment.platform.platform")
        def test_report_incompatible_environment(
            self, mock_platform, mock_python_version,
            mock_packages, env_manager
        ):
            """Test report for incompatible environment."""
            mock_python_version.return_value = "3.11.0"
            mock_platform.return_value = "Linux-5.15.0"
            mock_packages.return_value = {}

            saved_env = {
                "python": {"version": "3.10.0"},
                "system": {"platform": "Linux-5.15.0"},
                "packages": {
                    "numpy": {"version": "1.24.0", "is_critical": True},
                }
            }

            report = env_manager.generate_environment_report(saved_env)

            assert "ENVIRONMENT COMPATIBILITY REPORT" in report
            assert "RECOMMENDATION" in report

    # -------------------------------------------------------------------------
    # Helper Method Tests
    # -------------------------------------------------------------------------

    class TestHelperMethods:
        """Tests for helper methods."""

        @pytest.fixture
        def env_manager(self, tmp_path):
            return EnvironmentManager(tmp_path)

        def test_process_python_version_info_match(self, env_manager):
            """Test Python version info processing for matching versions."""
            with patch(
                "brisk.cli.environment.platform.python_version",
                return_value="3.10.0"
            ):
                saved_env = {"python": {"version": "3.10.0"}}
                info = env_manager._process_python_version_info(saved_env)

                assert info["saved"] == "3.10.0"
                assert info["current"] == "3.10.0"
                assert info["match"] is True
                assert info["major_minor_match"] is True

        def test_process_python_version_info_minor_mismatch(self, env_manager):
            """Test Python version info processing for minor mismatch."""
            with patch(
                "brisk.cli.environment.platform.python_version",
                return_value="3.11.0"
            ):
                saved_env = {"python": {"version": "3.10.0"}}
                info = env_manager._process_python_version_info(saved_env)

                assert info["match"] is False
                assert info["major_minor_match"] is False

        def test_process_system_info(self, env_manager):
            """Test system info processing."""
            with patch(
                "brisk.cli.environment.platform.platform",
                return_value="Linux-5.15.0"
            ):
                saved_env = {"system": {"platform": "Linux-5.10.0"}}
                info = env_manager._process_system_info(saved_env)

                assert info["saved_platform"] == "Linux-5.10.0"
                assert info["current_platform"] == "Linux-5.15.0"
                assert info["has_system_info"] is True

        def test_process_system_info_missing(self, env_manager):
            """Test system info processing when system info is missing."""
            saved_env = {}
            info = env_manager._process_system_info(saved_env)

            assert info["has_system_info"] is False
