"""Integration tests for project.py"""

# def test_nested_project_root(self, tmp_path, monkeypatch):
#     """Test finding .briskconfig in parent directory"""
#     # Create project structure
#     project_root = tmp_path / "project"
#     nested_dir = project_root / "nested" / "deep" / "directory"
#     nested_dir.mkdir(parents=True)
#     
#     # Create .briskconfig in project root
#     (project_root / '.briskconfig').touch()
#     
#     # Create datasets directory
#     datasets_dir = project_root / 'datasets'
#     datasets_dir.mkdir()
#     (datasets_dir / 'test.csv').touch()
#     
#     # Change working directory to nested directory
#     monkeypatch.chdir(nested_dir)
#     find_project_root.cache_clear()  # Clear before this specific test
#     
#     group = ExperimentGroupFactory.simple(
#     # group = ExperimentGroup(
#         name="test_group",
#         workflow="regression_workflow",
#         datasets=["test.csv"],
#         algorithms=["linear"]
#     )
#     path, _ = group.dataset_paths[0]
#     assert path == project_root / 'datasets' / 'test.csv'
