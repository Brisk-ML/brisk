"""Unit tests for project.py"""
import pytest

from brisk.configuration.project import find_project_root 

class TestFindProjectRoot():
    def test_project_root_not_found(self, tmp_path, monkeypatch):
        """Test FileNotFoundError when .briskconfig is not found"""
        monkeypatch.chdir(tmp_path)
        find_project_root.cache_clear()  # Clear before this specific test
        
        with pytest.raises(
            FileNotFoundError, match="Could not find .briskconfig"
        ):
            find_project_root()

