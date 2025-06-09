from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict

from glotaran.builtin.io.yml.utils import write_dict
from glotaran.io import save_dataset
from glotaran.io import save_parameters
from glotaran.io.interface import SAVING_OPTIONS_DEFAULT
from glotaran.io.interface import SavingOptions
from glotaran.model.errors import GlotaranUserError
from glotaran.model.experiment_model import ExperimentModel  # noqa: TC001
from glotaran.optimization import OptimizationInfo  # noqa: TC001
from glotaran.optimization.objective import OptimizationResult  # noqa: TC001
from glotaran.parameter import Parameters  # noqa: TC001
from glotaran.project.scheme import Scheme  # noqa: TC001

if TYPE_CHECKING:
    from pathlib import Path


class Result(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    optimization_results: dict[str, OptimizationResult]
    scheme: Scheme
    optimization_info: OptimizationInfo
    initial_parameters: Parameters
    optimized_parameters: Parameters

    @property
    def experiments(self) -> dict[str, ExperimentModel]:
        return self.scheme.experiments

    def save(
        self,
        path: Path,
        options: SavingOptions = SAVING_OPTIONS_DEFAULT,
        *,
        allow_overwrite: bool = False,
    ) -> None:
        if path.is_file():
            msg = "Save path must be a folder."
            raise GlotaranUserError(msg)
        if path.exists() and not allow_overwrite:
            msg = "Save path already exists. Use allow_overwrite=True to overwrite."
            raise GlotaranUserError(msg)
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
        data_format = options.get("data_format", "nc")

        for label, data in self.optimization_results.items():
            dataset_path = data_path / f"{label}.{data_format}"
            result_dict["data"][label] = str(dataset_path)
            # TODO: Make saving options more granular on a per element base
            # if options.data_filter is not None:
            #     data = data[options.data_filter]
            save_dataset(data, dataset_path, allow_overwrite=allow_overwrite)

        optimization_history_path = path / "optimization_history.csv"
        result_dict["optimization_history"] = str(optimization_history_path)
        self.optimization_info.optimization_history.to_csv(optimization_history_path)

        parameter_format = options.get("parameter_format", "csv")
        parameters_initial_path = path / f"parameters_initial.{parameter_format}"
        result_dict["parameters_initial"] = str(parameters_initial_path)
        save_parameters(
            self.initial_parameters,
            parameters_initial_path,
            allow_overwrite=allow_overwrite,
        )

        parameters_optimized_path = path / f"parameters_optimized.{parameter_format}"
        result_dict["parameters_optimized"] = str(parameters_optimized_path)
        save_parameters(
            self.optimized_parameters,
            parameters_optimized_path,
            allow_overwrite=allow_overwrite,
        )

        result_path = path / "glotaran_result.yml"
        write_dict(result_dict, result_path)
