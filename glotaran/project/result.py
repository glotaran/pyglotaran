from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import SerializationInfo
from pydantic import ValidationInfo
from pydantic import field_serializer
from pydantic import field_validator

from glotaran.builtin.io.yml.utils import write_dict
from glotaran.io import load_scheme
from glotaran.io import save_dataset
from glotaran.io import save_parameters
from glotaran.io import save_scheme
from glotaran.io.interface import SAVING_OPTIONS_DEFAULT
from glotaran.io.interface import SavingOptions
from glotaran.model.errors import GlotaranUserError
from glotaran.model.experiment_model import ExperimentModel  # noqa: TC001
from glotaran.optimization import OptimizationInfo  # noqa: TC001
from glotaran.optimization.objective import OptimizationResult
from glotaran.parameter import Parameters  # noqa: TC001
from glotaran.project.scheme import Scheme
from glotaran.utils.pydantic_serde import ValidationInfoWithContext
from glotaran.utils.pydantic_serde import context_is_dict
from glotaran.utils.pydantic_serde import deserialize_parameters
from glotaran.utils.pydantic_serde import save_folder_from_info
from glotaran.utils.pydantic_serde import serialization_info_to_kwargs
from glotaran.utils.pydantic_serde import serialize_parameters

if TYPE_CHECKING:
    from pathlib import Path


def inject_saving_option_from_data_into_context(info: ValidationInfoWithContext) -> None:
    """Retrieve saving options from data and inject them into context.

    Parameters
    ----------
    info : ValidationInfoWithContext
        Validation information containing context and data.

    Returns
    -------
    ValidationInfo
        Updated validation information with injected saving options from data.
    """
    info.context["saving_options"] = info.context.get(
        "saving_options", SAVING_OPTIONS_DEFAULT
    ) | info.data.get("saving_options", {})


class Result(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    saving_options: SavingOptions = SAVING_OPTIONS_DEFAULT
    optimization_results: dict[str, OptimizationResult]
    scheme: Scheme
    optimization_info: OptimizationInfo
    initial_parameters: Parameters
    optimized_parameters: Parameters

    @property
    def experiments(self) -> dict[str, ExperimentModel]:
        return self.scheme.experiments

    @field_serializer("saving_options")
    def serialize_saving_options(
        self, value: SavingOptions, info: SerializationInfo
    ) -> SavingOptions:
        """Use saving options from context if available."""
        if context_is_dict(info):
            value |= info.context.get("saving_options", SAVING_OPTIONS_DEFAULT)
        return value

    @field_serializer("optimization_results", when_used="json")
    def serialize_optimization_results(
        self, value: dict[str, OptimizationResult], info: SerializationInfo
    ) -> dict[str, dict[str, Any]]:
        """Serialize the optimization results to a dictionary."""
        if context_is_dict(info) and (save_folder := save_folder_from_info(info)) is not None:
            return {
                dataset_name: optimization_result.model_dump(
                    mode="json",
                    context=info.context
                    | {"save_folder": save_folder / "optimization_results" / dataset_name},
                    **serialization_info_to_kwargs(info, exclude={"mode", "context"}),
                )
                for dataset_name, optimization_result in value.items()
            }
        msg = (
            "SerializationInfo context is missing 'save_folder' for "
            f"field '{info.field_name}':\n{info}"  # type: ignore[attr-defined]
        )
        raise ValueError(msg)

    @field_validator("optimization_results", mode="before")
    @classmethod
    def validate_optimization_results(cls, value: Any, info: ValidationInfo) -> Any:  # noqa: ANN401
        """Validate the data field."""
        if context_is_dict(info) and (save_folder := save_folder_from_info(info)) is not None:
            inject_saving_option_from_data_into_context(info)
            return {
                dataset_name: OptimizationResult.model_validate(
                    optimization_result,
                    context=info.context
                    | {"save_folder": save_folder / "optimization_results" / dataset_name},
                )
                for dataset_name, optimization_result in value.items()
            }
        return value

    @field_serializer("initial_parameters", "optimized_parameters", when_used="json")
    def serialize_parameters(self, value: Parameters, info: SerializationInfo) -> Any:  # noqa: ANN401
        """Serialize the parameter fields."""
        return serialize_parameters(value, info)

    @field_validator("initial_parameters", "optimized_parameters", mode="before")
    @classmethod
    def validate_parameters(cls, value: Any, info: ValidationInfo) -> Any:  # noqa: ANN401
        """Validate parameters fields."""
        if context_is_dict(info):
            inject_saving_option_from_data_into_context(info)
        return deserialize_parameters(value, info)

    @field_serializer("scheme", when_used="json")
    def serialize_scheme(self, value: Scheme, info: SerializationInfo) -> Any:  # noqa: ANN401
        """Serialize the scheme field."""
        assert context_is_dict(info)
        if (save_folder := save_folder_from_info(info)) is not None:
            scheme_format = info.context.get("saving_options", {}).get("scheme_format", "yml")
            save_path = save_folder / f"scheme.{scheme_format}"
            save_scheme(value, save_path)
            return save_path.name
        msg = (
            "SerializationInfo context is missing 'save_folder' for "
            f"field '{info.field_name}':\n{info}"
        )
        raise ValueError(msg)

    @field_validator("scheme", mode="before")
    @classmethod
    def validate_scheme(cls, value: Any, info: ValidationInfo) -> Any:  # noqa: ANN401
        """Initialize scheme from dict or string if necessary."""
        if isinstance(value, dict) is True:
            return Scheme.from_dict(value)
        if isinstance(value, str) is True:
            assert context_is_dict(info)
            inject_saving_option_from_data_into_context(info)
            scheme_plugin = info.context.get("saving_options", {}).get("scheme_plugin", None)
            if (save_folder := save_folder_from_info(info)) is not None:
                value = save_folder / value
            return load_scheme(value, format_name=scheme_plugin)
        return value

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
