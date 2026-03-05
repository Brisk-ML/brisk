"""Integration tests for find_project_root."""
import pytest

from brisk.configuration import project

@pytest.fixture(autouse=True)
def clear_root_cache():
    """Clear the lru_cache before every test to ensure isolation."""
    project.find_project_root.cache_clear()


@pytest.mark.integration
class TestFindProjectRootIntegration:
    """Integration tests for the find_project_root function."""

    def test_in_current_dir(self, tmp_path, monkeypatch):
        config_file = tmp_path / ".briskconfig"
        config_file.touch()
        monkeypatch.chdir(tmp_path)

        assert project.find_project_root() == tmp_path

    def test_in_parent_dir(self, tmp_path, monkeypatch):
        config_file = tmp_path / ".briskconfig"
        config_file.touch()

        child_dir = tmp_path / "subdir" / "nested_more"
        child_dir.mkdir(parents=True)

        monkeypatch.chdir(child_dir)

        assert project.find_project_root() == tmp_path

    def test_in_child_dir(self, tmp_path, monkeypatch):
        child_dir = tmp_path / "child"
        child_dir.mkdir()
        (child_dir / ".briskconfig").touch()

        monkeypatch.chdir(tmp_path)

        with pytest.raises(
            FileNotFoundError, match="Could not find a brisk project"
        ):
            project.find_project_root()
