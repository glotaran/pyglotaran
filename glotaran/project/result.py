from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict

from glotaran.builtin.io.yml.utils import write_dict
from glotaran.io import save_dataset
from glotaran.io import save_parameters
from glotaran.model.errors import GlotaranUserError
from glotaran.model.experiment_model import ExperimentModel  # noqa: TCH001
from glotaran.optimization import OptimizationResult  # noqa: TCH001
from glotaran.optimization.objective import DatasetResult
from glotaran.parameter import Parameters  # noqa: TCH001

if TYPE_CHECKING:
    from pathlib import Path


class SavingOptions(BaseModel):
    """A collection of options for result saving."""

    data_filter: list[str] | None = None
    data_format: Literal["nc"] = "nc"
    parameter_format: Literal["csv"] = "csv"


SAVING_OPTIONS_DEFAULT = SavingOptions()


class Result(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    datasets: dict[str, DatasetResult]
    experiments: dict[str, ExperimentModel]
    optimization: OptimizationResult
    parameters_intitial: Parameters
    parameters_optimized: Parameters

    def save(
        self,
        path: Path,
        options: SavingOptions = SAVING_OPTIONS_DEFAULT,
        allow_overwrite: bool = False,
    ):
        if path.is_file():
            raise GlotaranUserError("Save path must be a folder.")
        if path.exists() and not allow_overwrite:
            raise GlotaranUserError(
                "Save path already exists. Use allow_overwrite=True to overwrite."
            )
        result_dict: dict[str, Any] = {"data": {}, "experiments": {}}
        path.mkdir(exist_ok=True, parents=True)

        # TODO: Save scheme or experiments
        #  experiment_folder = path / "experiments"
        #  experiment_folder.mkdir()
        #  for label, experiment in self.experiments.items():
        #      experiment_path = experiment_folder / f"{label}.yml"
        #      result_dict["experiments"][label] = experiment_path
        #      write_dict(experiment.model_dump(), experiment_path)

        data_path = path / "data"
        data_path.mkdir(exist_ok=True)
        for label, data in self.datasets.items():
            dataset_path = data_path / f"{label}.{options.data_format}"
            result_dict["data"][label] = str(dataset_path)
            if options.data_filter is not None:
                data = data[options.data_filter]
            save_dataset(data, dataset_path, allow_overwrite=allow_overwrite)

        optimization_history_path = path / "optimization_history.csv"
        result_dict["optimization_history"] = str(optimization_history_path)
        self.optimization.optimization_history.to_csv(optimization_history_path)

        parameters_initial_path = path / f"parameters_initial.{options.parameter_format}"
        result_dict["parameters_initial"] = str(parameters_initial_path)
        save_parameters(
            self.parameters_intitial,
            parameters_initial_path,
            allow_overwrite=allow_overwrite,
        )

        parameters_optimized_path = path / f"parameters_optimized.{options.parameter_format}"
        result_dict["parameters_optimized"] = str(parameters_optimized_path)
        save_parameters(
            self.parameters_optimized,
            parameters_optimized_path,
            allow_overwrite=allow_overwrite,
        )

        result_path = path / "glotaran_result.yml"
        write_dict(result_dict, result_path)
