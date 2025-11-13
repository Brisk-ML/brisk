"""Factories to create objects for testing."""
from typing import Dict, Any
from pathlib import Path

from sklearn import linear_model

from brisk.configuration.algorithm_wrapper import AlgorithmWrapper
from brisk.configuration.experiment_group import ExperimentGroup
from brisk.configuration.algorithm_collection import AlgorithmCollection

class AlgorithmFactory:
    """Factory to create AlgorithmWrapper instances for use in tests."""
    @classmethod
    def simple(cls):
        return AlgorithmWrapper(
            name="ridge",
            display_name="Ridge Regression",
            algorithm_class=linear_model.Ridge,
            default_params={"alpha": 0.5},
            hyperparam_grid={"alpha": [0.1, 0.5, 1.0]}
        )

    @classmethod
    def full(
        cls,
        name: str = "ridge",
        display_name: str = "Ridge Regression",
        algorithm_class=linear_model.Ridge,
        default_params: Dict[str, Any] = {"alpha": 0.5},
        hyperparam_grid: Dict[str, Any] = {"alpha": [0.1, 0.5, 1.0]}
    ):
        return AlgorithmWrapper(
            name=name,
            display_name=display_name,
            algorithm_class=algorithm_class,
            default_params=default_params,
            hyperparam_grid=hyperparam_grid
        )

    @classmethod
    def collection(cls):
        return AlgorithmCollection(
            cls.simple(),
            cls.full(
                "linear", "Linear Regression", linear_model.LinearRegression,
                {}, {}
            ),
            cls.full("lasso", "LASSO Regression", linear_model.Lasso)
        )

class ExperimentGroupFactory:
    DEFAULT_NAME = "test_group"
    DEFAULT_DATASETS = ["data.csv"]
    DEFAULT_WORKFLOW = "test_workflow"
    DEFAULT_ALGORITHMS = ["ridge"]

    @classmethod
    def _create_dataset_files(cls, tmp_path: Path, datasets: list[str]):
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        for dataset in datasets:
            if isinstance(dataset, str):
                (datasets_dir / dataset).write_text("")
            elif isinstance(dataset, tuple):
                (datasets_dir / dataset[0]).write_text("")

    @classmethod
    def simple(
        cls,
        tmp_path: Path,
        name: str | None = None,
        datasets: list[str] | None = None,
        workflow: str | None = None,
        algorithms: list[str] | None = None,
        create_files: bool = True
    ):
        if create_files:
            cls._create_dataset_files(
                tmp_path, datasets or cls.DEFAULT_DATASETS
            )
        return ExperimentGroup(
            name=name or cls.DEFAULT_NAME,
            datasets=datasets or cls.DEFAULT_DATASETS,
            workflow=workflow or cls.DEFAULT_WORKFLOW,
            algorithms=algorithms or cls.DEFAULT_ALGORITHMS
        )
    
    @classmethod
    def with_data_config(
        cls,
        tmp_path: Path,
        data_config: dict,
        name: str | None = None,
        datasets: list[str] | None = None,
        workflow: str | None = None,
        algorithms: list[str] | None = None,
        create_files: bool = True
    ):
        if create_files:
            cls._create_dataset_files(
                tmp_path, datasets or cls.DEFAULT_DATASETS
            )
        return ExperimentGroup(
            name=name or cls.DEFAULT_NAME,
            data_config=data_config,
            datasets=datasets or cls.DEFAULT_DATASETS,
            workflow=workflow or cls.DEFAULT_WORKFLOW,
            algorithms=algorithms or cls.DEFAULT_ALGORITHMS
        )


    @classmethod
    def with_hyperparam_grid(
        cls,
        tmp_path: Path,
        hyperparam_grid: dict,
        name: str | None = None,
        datasets: list[str] | None = None,
        workflow: str | None = None,
        algorithms: list[str] | None = None,
        create_files: bool = True
    ):
        if create_files:
            cls._create_dataset_files(
                tmp_path, datasets or cls.DEFAULT_DATASETS
            )
        return ExperimentGroup(
            name=name or cls.DEFAULT_NAME,
            algorithm_config=hyperparam_grid,
            datasets=datasets or cls.DEFAULT_DATASETS,
            workflow=workflow or cls.DEFAULT_WORKFLOW,
            algorithms=algorithms or cls.DEFAULT_ALGORITHMS
        )

    @classmethod
    def with_description(
        cls,
        tmp_path: Path,
        description: str,
        name: str | None = None,
        datasets: list[str] | None = None,
        workflow: str | None = None,
        create_files: bool = True
    ):
        if create_files:
            cls._create_dataset_files(
                tmp_path, datasets or cls.DEFAULT_DATASETS
            )
        return ExperimentGroup(
            description=description,
            name=name or cls.DEFAULT_NAME,
            datasets=datasets or cls.DEFAULT_DATASETS,
            workflow=workflow or cls.DEFAULT_WORKFLOW
        )
