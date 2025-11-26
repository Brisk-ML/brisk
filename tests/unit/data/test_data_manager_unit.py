"""Unit tests for DataManager."""

import pytest
from sklearn import model_selection

from brisk.data.data_manager import DataManager
from brisk.data.preprocessing import (
    ScalingPreprocessor, MissingDataPreprocessor
)

class TestDataManagerUnit:
    def test_initalization(self):
        manager = DataManager(
            test_size=0.3,
            n_splits=3,
            split_method="shuffle",
            group_column="group",
            stratified=False,
            random_state=42,
            problem_type="regression",
            algorithm_config=None,
            preprocessors=None
        )
        assert manager.preprocessors == []

    def test_validate_config_invalid_split_method(self):
        with pytest.raises(ValueError):
            manager = DataManager(
                test_size=0.3,
                n_splits=3,
                split_method="not_valid_option",
                group_column="group",
                stratified=False,
                random_state=42,
                problem_type="regression",
                algorithm_config=None,
                preprocessors=None
            )

    def test_validate_config_stratified_shuffle_invalid(self):
        with pytest.raises(ValueError):
            manager = DataManager(
                test_size=0.3,
                n_splits=3,
                split_method="shuffle",
                group_column="group",
                stratified=True,
                random_state=42,
                problem_type="regression",
                algorithm_config=None,
                preprocessors=None
            )

    def test_validate_config_invalid_problem_type(self):
        with pytest.raises(ValueError):
            manager = DataManager(
                test_size=0.3,
                n_splits=3,
                split_method="shuffle",
                group_column="group",
                stratified=False,
                random_state=42,
                problem_type="not_valid_option",
                algorithm_config=None,
                preprocessors=None
            )

    def test_group_shuffle_split(self):
        manager = DataManager(
            test_size=0.3,
            n_splits=3,
            split_method="shuffle",
            group_column="group",
            stratified=False,
            random_state=42,
            problem_type="regression",
            algorithm_config=None,
            preprocessors=None
        )
        assert isinstance(manager.splitter, model_selection.GroupShuffleSplit)

    def test_stratified_shuffle_split(self):
        manager = DataManager(
            test_size=0.3,
            n_splits=3,
            split_method="shuffle",
            group_column=None,
            stratified=True,
            random_state=42,
            problem_type="regression",
            algorithm_config=None,
            preprocessors=None
        )
        assert isinstance(
            manager.splitter, model_selection.StratifiedShuffleSplit
        )

    def test_shuffle_split(self):
        manager = DataManager(
            test_size=0.3,
            n_splits=3,
            split_method="shuffle",
            group_column=None,
            stratified=False,
            random_state=42,
            problem_type="regression",
            algorithm_config=None,
            preprocessors=None
        )
        assert isinstance(manager.splitter, model_selection.ShuffleSplit)

    def test_group_kfold_split(self):
        manager = DataManager(
            test_size=0.3,
            n_splits=3,
            split_method="kfold",
            group_column="group",
            stratified=False,
            random_state=42,
            problem_type="regression",
            algorithm_config=None,
            preprocessors=None
        )
        assert isinstance(manager.splitter, model_selection.GroupKFold)

    def test_stratified_kfold_split(self):
        manager = DataManager(
            test_size=0.3,
            n_splits=3,
            split_method="kfold",
            group_column=None,
            stratified=True,
            random_state=42,
            problem_type="regression",
            algorithm_config=None,
            preprocessors=None
        )
        assert isinstance(manager.splitter, model_selection.StratifiedKFold)

    def test_kfold_split(self):
        manager = DataManager(
            test_size=0.3,
            n_splits=3,
            split_method="kfold",
            group_column=None,
            stratified=False,
            random_state=42,
            problem_type="regression",
            algorithm_config=None,
            preprocessors=None
        )
        assert isinstance(manager.splitter, model_selection.KFold)

    def test_stratified_group_kfold_split(self):
        manager = DataManager(
            test_size=0.3,
            n_splits=3,
            split_method="kfold",
            group_column="group",
            stratified=True,
            random_state=42,
            problem_type="regression",
            algorithm_config=None,
            preprocessors=None
        )
        assert isinstance(
            manager.splitter, model_selection.StratifiedGroupKFold
        )

    def test_invalid_splitter_combination(self):
        with pytest.raises(ValueError):
            manager = DataManager(
                test_size=0.3,
                n_splits=3,
                split_method="shuffle",
                group_column="group",
                stratified=True,
                random_state=42,
                problem_type="regression",
                algorithm_config=None,
                preprocessors=None
            )

    def test_export_params_no_preprocessor(self):
        input_params = {
            "test_size":0.3,
            "n_splits":3,
            "split_method":"shuffle",
            "group_column":"group",
            "stratified":False,
            "random_state":42,
            "problem_type":"regression",
        }
        manager = DataManager(
            **input_params,
            preprocessors=None
        )
        json = manager.export_params()
        
        input_params["preprocessors"] = {}
        assert json == {"params": input_params}
        
    def test_export_params_one_preprocessor(self):
        input_params = {
            "test_size":0.3,
            "n_splits":3,
            "split_method":"shuffle",
            "group_column":"group",
            "stratified":False,
            "random_state":42,
            "problem_type":"regression",
        }
        manager = DataManager(
            **input_params,
            preprocessors=[ScalingPreprocessor(method="minmax")]
        )
        json = manager.export_params()

        input_params["preprocessors"] = {
            "ScalingPreprocessor": {"method": "minmax"}
        }
        assert json == {"params": input_params}

    def test_export_params_two_preprocessor(self):
        input_params = {
            "test_size":0.3,
            "n_splits":3,
            "split_method":"shuffle",
            "group_column":"group",
            "stratified":False,
            "random_state":42,
            "problem_type":"regression",
        }
        manager = DataManager(
            **input_params,
            preprocessors=[
                ScalingPreprocessor(method="minmax"),
                MissingDataPreprocessor(
                    strategy="drop_rows",
                    impute_method="mean",
                    constant_value=1 
                )
            ]
        )
        json = manager.export_params()

        input_params["preprocessors"] = {
            "ScalingPreprocessor": {"method": "minmax"},
            "MissingDataPreprocessor": {
                "strategy": "drop_rows",
                "impute_method": "mean",
                "constant_value": 1
            }
        }
        assert json == {"params": input_params}
