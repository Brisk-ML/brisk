"""Integration tests for ConfigurationManager."""
#  def test_two_data_managers(self, mock_brisk_project):
#         """Test correct data manager is loaded if multiple are defined"""
#         data_file = mock_brisk_project / "data.py"
#         data_file.unlink()
#         data_content = textwrap.dedent("""
#             from brisk.data.data_manager import DataManager
#             BASE_DATA_MANAGER = DataManager(
#                 test_size=0.2,
#                 n_splits=5,
#                 split_method="shuffle"
#             )
#             BASE_DATA_MANAGER = DataManager(
#                 test_size=0.3,
#                 n_splits=5,
#                 split_method="kfold"
#             )
#         """).strip()
#         data_file.write_text(data_content)
#         group = ExperimentGroup(
#             name="test_group",
#             workflow="regression_workflow",
#             datasets=["regression.csv"],
#             algorithms=["linear"]
#         )
#         with pytest.raises(
#             ValueError, 
#             match="BASE_DATA_MANAGER is defined multiple times in"
#             ):
#             manager = ConfigurationManager([group], {}, PlotSettings())

#     def test_base_data_manager_wrong_class(self, mock_brisk_project):
#         """Test error handling for invalid base data manager class"""
#         data_file = mock_brisk_project / "data.py"
#         data_file.unlink()
#         data_content = textwrap.dedent("""
#             from brisk.data.data_manager import DataManager
#             BASE_DATA_MANAGER = 1
#         """).strip()
#         data_file.write_text(data_content)
#         with pytest.raises(
#             ValueError, 
#             match="is not a valid DataManager instance"
#             ):
#             manager = ConfigurationManager([], {}, PlotSettings())
    
#     def test_validate_single_data_manager_two_definitions(
#             self,
#             mock_brisk_project
#         ):
#         """Test the _validate_single_data_manager method error handling."""
#         data_file = mock_brisk_project / "data.py"
#         data_file.unlink()
#         data_content = textwrap.dedent("""
#             from brisk.data.data_manager import DataManager
#             BASE_DATA_MANAGER = DataManager(
#                 test_size=0.2,
#                 n_splits=5,
#                 split_method="shuffle"
#             )
#             BASE_DATA_MANAGER = DataManager(
#                 test_size=0.3,
#                 n_splits=5,
#                 split_method="kfold"
#             )
#         """).strip()
#         data_file.write_text(data_content)

#         with pytest.raises(
#             ValueError, 
#             match="BASE_DATA_MANAGER is defined multiple times in"
#             ):
#             manager = ConfigurationManager([], {}, PlotSettings())

#     def test_validate_single_data_manager_invalid_syntax(
#             self,
#             mock_brisk_project
#         ):
#         """Test the _validate_single_data_manager method error handling."""
#         data_file = mock_brisk_project / "data.py"
#         data_file.unlink()
#         data_content = textwrap.dedent("""
#             from brisk.data.data_manager import DataManager
#             BASE_DATA_MANAGER = DataManager(
#                 test_size=0.2
#                 n_splits=5
#                 split_method="shuffle"
#             )
#         """).strip()
#         data_file.write_text(data_content)

#         with pytest.raises(SyntaxError, match="invalid syntax"):
#             manager = ConfigurationManager([], {}, PlotSettings())

    # def test_missing_algorithm_file(self, mock_brisk_project):
    #     """Test error handling for missing algorithms.py."""
    #     algorithm_file = mock_brisk_project / 'algorithms.py'
    #     algorithm_file.unlink()
    #     group = ExperimentGroup(
    #         name="test",
    #         workflow="regression_workflow",
    #         datasets=["regression.csv"],
    #         algorithms=["linear"]
    #     )
        
    #     with pytest.raises(
    #         FileNotFoundError, 
    #         match="algorithms.py file not found:"
    #         ):
    #         ConfigurationManager([group], {}, PlotSettings())

    # def test_missing_algorithm_config(self, mock_brisk_project):
    #     """Test error handling for algorithms.py without ALGORITHM_CONFIG."""
    #     algorithm_file = mock_brisk_project / 'algorithms.py'
    #     algorithm_file.unlink()
    #     algorithm_content = textwrap.dedent("""
    #         # Missing ALGORITHM_CONFIG
    #     """).strip()
    #     algorithm_file.write_text(algorithm_content)
    #     group = ExperimentGroup(
    #         name="test",
    #         workflow="regression_workflow",
    #         datasets=["regression.csv"],
    #         algorithms=["linear"]
    #     )
        
    #     with pytest.raises(ImportError, match="ALGORITHM_CONFIG not found in"):
    #         ConfigurationManager([group], {}, PlotSettings())
      
    # def test_invalid_algorithm_file(self, mock_brisk_project):
    #     """Test error handling for invalid algorithms.py file."""
    #     algorithm_file = mock_brisk_project / 'algorithms.py'
    #     algorithm_file.unlink()
    #     algorithm_content = textwrap.dedent("""
    #         from brisk.configuration.algorithm_collection import AlgorithmCollection
    #         algorithm_config = AlgorithmCollection()
    #     """).strip()
    #     algorithm_file.write_text(algorithm_content)

    #     with pytest.raises(
    #         ImportError, 
    #         match="ALGORITHM_CONFIG not found in"
    #         ):
    #         manager = ConfigurationManager([], {}, PlotSettings())

    # def test_unloadable_algorithm_file(self, mock_brisk_project, monkeypatch):
    #     """Test error handling for invalid data.py file."""
    #     original_spec_from_file_location = importlib.util.spec_from_file_location

    #     def mock_spec_from_file_location(name, location, *args, **kwargs):
    #         if name == 'algorithms': # If module is 'algorithms.py', return None
    #             return None
    #         return original_spec_from_file_location( # pragma: no cover
    #             name, location, *args, **kwargs
    #             )

    #     monkeypatch.setattr(
    #         'importlib.util.spec_from_file_location', 
    #         mock_spec_from_file_location
    #         )
    #     group = ExperimentGroup(
    #         name="test_group",
    #         workflow="regression_workflow",
    #         datasets=["regression.csv"],
    #         algorithms=["linear"]
    #     )
        
    #     with pytest.raises(ImportError, match="Failed to load algorithms module"):
    #         ConfigurationManager([group], {}, PlotSettings())

    # def test_validate_two_algorithm_configs(self, mock_brisk_project):
    #     """Test two ALGORITHM_CONFIGs are not allowed."""
    #     algorithm_file = mock_brisk_project / 'algorithms.py'
    #     algorithm_file.unlink()
    #     algorithm_content = textwrap.dedent("""
    #         from brisk.configuration.algorithm_collection import AlgorithmCollection
    #         ALGORITHM_CONFIG = AlgorithmCollection()
    #         ALGORITHM_CONFIG = AlgorithmCollection()
    #     """).strip()
    #     algorithm_file.write_text(algorithm_content)
    #     with pytest.raises(
    #         ValueError, 
    #         match="ALGORITHM_CONFIG is defined multiple times in"
    #         ):
    #         manager = ConfigurationManager([], {}, PlotSettings())

    # def test_single_algorithm_config_invalid_syntax(
    #         self,
    #         mock_brisk_project
    #     ):
    #     """Test error handling for invalid ALGORITHM_CONFIG syntax."""
    #     algorithm_file = mock_brisk_project / 'algorithms.py'
    #     algorithm_file.unlink()
    #     algorithm_content = textwrap.dedent("""
    #         from brisk.configuration.algorithm_collection import AlgorithmCollection
    #         from brisk.configuration.algorithm_wrapper import AlgorithmWrapper
    #         from sklearn.linear_model import LinearRegression
    #         ALGORITHM_CONFIG = AlgorithmCollection(
    #             AlgorithmWrapper(
    #                 name="linear"
    #                 display_name="Linear Regression"
    #                 algorithm_class=LinearRegression
    #             )
    #         )
    #     """).strip()
    #     algorithm_file.write_text(algorithm_content)

    #     with pytest.raises(SyntaxError, match="invalid syntax"):
    #         manager = ConfigurationManager([], {}, PlotSettings())

#     def test_create_logfile(self, base_data_manager):
#         """Test the _create_logfile method of ConfigurationManager."""
#         group1 = ExperimentGroupFactory.simple(
#             name="group1",
#             workflow="regression_workflow",
#             datasets=["regression.csv"],
#             algorithms=["linear"]
#         )
#
#         group2 = ExperimentGroupFactory.simple(
#             name="group2",
#             workflow="regression_workflow",
#             datasets=["regression.csv"],
#             algorithms=["ridge"],
#             algorithm_config={"ridge": {"alpha": 0.5}}
#         )
#         
#         manager = ConfigurationManager([group1, group2], {}, PlotSettings())
#         manager.base_data_manager = base_data_manager
#         manager.algorithm_config = AlgorithmFactory.collection()
#         manager.get_data_managers()
#         manager.create_logfile()
#
#         expected_logfile_content = """
# ## Default Algorithm Configuration
# ### Linear Regression (`linear`)
#
# - **Algorithm Class**: `LinearRegression`
#
# **Default Parameters:**
# ```python
# {}
# ```
#
# **Hyperparameter Grid:**
# ```python
# {}
# ```
#
# ### Ridge Regression (`ridge`)
#
# - **Algorithm Class**: `Ridge`
#
# **Default Parameters:**
# ```python
# 'max_iter': 10000,
# ```
#
# **Hyperparameter Grid:**
# ```python
# 'alpha': [0.1, 0.5, 1.0],
# ```
#
# ### Elastic Net Regression (`elasticnet`)
#
# - **Algorithm Class**: `ElasticNet`
#
# **Default Parameters:**
# ```python
# 'alpha': 0.1,
# 'max_iter': 10000,
# ```
#
# **Hyperparameter Grid:**
# ```python
# 'alpha': [0.1, 0.2, 0.5],
# 'l1_ratio': [0.1, 0.5, 1.0],
# ```
#
# ### Random Forest (`rf`)
#
# - **Algorithm Class**: `RandomForestRegressor`
#
# **Default Parameters:**
# ```python
# 'n_jobs': 1,
# ```
#
# **Hyperparameter Grid:**
# ```python
# 'n_estimators': [20, 40, 60, 80, 100, 120, 140],
# ```
#
# ### Random Forest Classifier (`rf_classifier`)
#
# - **Algorithm Class**: `RandomForestClassifier`
#
# **Default Parameters:**
# ```python
# 'min_samples_split': 10,
# ```
#
# **Hyperparameter Grid:**
# ```python
# 'n_estimators': [20, 40, 60, 80, 100, 120, 140],
# 'criterion': ['friedman_mse', 'absolute_error', 'poisson', 'squared_error'],
# 'max_depth': [5, 10, 15, 20, None],
# ```
#
# ## Experiment Group: group1
# #### Description: 
#
# ### DataManager Configuration
# ```python
# DataManager Configuration:
# test_size: 0.2
# n_splits: 2
# split_method: shuffle
# stratified: False
# random_state: 42
# problem_type: classification
# preprocessors: []
# ```
#
# ### Datasets
# #### regression.csv
# Features:
# ```python
# Categorical: []
# Continuous: ['x', 'y']
# ```
#
# ## Experiment Group: group2
# #### Description: 
#
# ### Algorithm Configurations
# ```python
# 'ridge': {'alpha': 0.5},
# ```
#
# ### DataManager Configuration
# ```python
# DataManager Configuration:
# test_size: 0.2
# n_splits: 2
# split_method: shuffle
# stratified: False
# random_state: 42
# problem_type: classification
# preprocessors: []
# ```
#
# ### Datasets
# #### regression.csv
# Features:
# ```python
# Categorical: []
# Continuous: ['x', 'y']
# ```
# """
#         # Strip whitespace from both files
#         print("ACTUAL LOGFILE CONTENT:")
#         print(manager.logfile)
#         print("EXPECTED LOGFILE CONTENT:")
#         print(expected_logfile_content)
#         assert manager.logfile.strip() == expected_logfile_content.strip()

#     def test_create_logfile_with_all_args(self, mock_brisk_project):
#         """Test the _create_logfile method of ConfigurationManager."""
#         group1 = ExperimentGroup(
#             name="group1",
#             workflow="regression_workflow",
#             datasets=["regression.csv", "categorical.csv"],
#             data_config={
#                 "test_size": 0.3,
#                 "n_splits": 3
#             },
#             algorithms=["linear", "ridge", "elasticnet"],
#             algorithm_config={
#                 "ridge": {"alpha": [0.1, 0.2, 0.5, 0.7, 0.9]},
#                 "elasticnet": {
#                     "alpha": [0.1, 0.2, 0.6, 0.75, 0.95],
#                     "l1_ratio": [0.01, 0.05, 0.1, 0.5, 0.8]
#                 }
#             },
#             description="This is a test description of group1",
#             workflow_args={
#                 "kfold": 2,
#                 "metrics": ["MAE", "R2"]
#             }
#         )
#
#         group2 = ExperimentGroup(
#             name="group2",
#             workflow="regression_workflow",
#             datasets=["group.csv"],
#             data_config={
#                 "test_size": 0.1,
#                 "split_method": "shuffle",
#                 "group_column": "group"
#             },
#             algorithms=["ridge", "elasticnet"],
#             algorithm_config={"ridge": {"alpha": [0.5, 0.3, 0.1]}},
#             description="This describes group2 in great detail",
#             workflow_args={
#                 "kfold": 3,
#                 "metrics": ["MSE", "CCC"]
#             }
#         )
#         
#         manager = ConfigurationManager([group1, group2], {
#             "categorical.csv": ["category"]
#         }, PlotSettings())
#         manager._create_logfile()
#
#         expected_logfile_content = """
# ## Default Algorithm Configuration
# ### Linear Regression (`linear`)
#
# - **Algorithm Class**: `LinearRegression`
#
# **Default Parameters:**
# ```python
# {}
# ```
#
# **Hyperparameter Grid:**
# ```python
# {}
# ```
#
# ### Ridge Regression (`ridge`)
#
# - **Algorithm Class**: `Ridge`
#
# **Default Parameters:**
# ```python
# 'max_iter': 10000,
# ```
#
# **Hyperparameter Grid:**
# ```python
# 'alpha': [0.1, 0.5, 1.0],
# ```
#
# ### Elastic Net Regression (`elasticnet`)
#
# - **Algorithm Class**: `ElasticNet`
#
# **Default Parameters:**
# ```python
# 'alpha': 0.1,
# 'max_iter': 10000,
# ```
#
# **Hyperparameter Grid:**
# ```python
# 'alpha': [0.1, 0.2, 0.5],
# 'l1_ratio': [0.1, 0.5, 1.0],
# ```
#
# ### Random Forest (`rf`)
#
# - **Algorithm Class**: `RandomForestRegressor`
#
# **Default Parameters:**
# ```python
# 'n_jobs': 1,
# ```
#
# **Hyperparameter Grid:**
# ```python
# 'n_estimators': [20, 40, 60, 80, 100, 120, 140],
# ```
#
# ### Random Forest Classifier (`rf_classifier`)
#
# - **Algorithm Class**: `RandomForestClassifier`
#
# **Default Parameters:**
# ```python
# 'min_samples_split': 10,
# ```
#
# **Hyperparameter Grid:**
# ```python
# 'n_estimators': [20, 40, 60, 80, 100, 120, 140],
# 'criterion': ['friedman_mse', 'absolute_error', 'poisson', 'squared_error'],
# 'max_depth': [5, 10, 15, 20, None],
# ```
#
# ## Experiment Group: group1
# #### Description: This is a test description of group1
#
# ### Algorithm Configurations
# ```python
# 'ridge': {'alpha': [0.1, 0.2, 0.5, 0.7, 0.9]},
# 'elasticnet': {'alpha': [0.1, 0.2, 0.6, 0.75, 0.95], 'l1_ratio': [0.01, 0.05, 0.1, 0.5, 0.8]},
# ```
#
# ### DataManager Configuration
# ```python
# DataManager Configuration:
# test_size: 0.3
# n_splits: 3
# split_method: shuffle
# stratified: False
# random_state: 42
# problem_type: classification
# preprocessors: []
# ```
#
# ### Datasets
# #### regression.csv
# Features:
# ```python
# Categorical: []
# Continuous: ['x', 'y']
# ```
#
# #### categorical.csv
# Features:
# ```python
# Categorical: ['category']
# Continuous: ['value']
# ```
#
# ## Experiment Group: group2
# #### Description: This describes group2 in great detail
#
# ### Algorithm Configurations
# ```python
# 'ridge': {'alpha': [0.5, 0.3, 0.1]},
# ```
#
# ### DataManager Configuration
# ```python
# DataManager Configuration:
# test_size: 0.1
# n_splits: 2
# split_method: shuffle
# group_column: group
# stratified: False
# random_state: 42
# problem_type: classification
# preprocessors: []
# ```
#
# ### Datasets
# #### group.csv
# Features:
# ```python
# Categorical: []
# Continuous: ['x', 'y']
# ```
# """
#         # Strip whitespace from both files
#         print("ACTUAL LOGFILE CONTENT:")
#         print(manager.logfile)
#         print("EXPECTED LOGFILE CONTENT:")
#         print(expected_logfile_content)
#         assert manager.logfile.strip() == expected_logfile_content.strip()
#
