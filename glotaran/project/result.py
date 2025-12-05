from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import SerializationInfo
from pydantic import ValidationInfo
from pydantic import field_serializer
from pydantic import field_validator

from glotaran.io import load_scheme
from glotaran.io import save_result
from glotaran.io import save_scheme
from glotaran.io.interface import SAVING_OPTIONS_DEFAULT
from glotaran.io.interface import SavingOptions
from glotaran.model.experiment_model import ExperimentModel  # noqa: TC001
from glotaran.optimization import OptimizationInfo  # noqa: TC001
from glotaran.optimization.objective import OptimizationResult
from glotaran.parameter import Parameters  # noqa: TC001
from glotaran.project.scheme import Scheme
from glotaran.utils.io import relative_posix_path
from glotaran.utils.pydantic_serde import ValidationInfoWithContext
from glotaran.utils.pydantic_serde import context_is_dict
from glotaran.utils.pydantic_serde import deserialize_parameters
from glotaran.utils.pydantic_serde import save_folder_from_info
from glotaran.utils.pydantic_serde import serialization_info_to_kwargs
from glotaran.utils.pydantic_serde import serialize_parameters

if TYPE_CHECKING:
    import xarray as xr


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

    @property
    def input_data(self) -> dict[str, xr.Dataset | xr.DataArray]:
        """Input data used to create the result."""
        return {
            dataset_name: optimization_result.input_data
            for dataset_name, optimization_result in self.optimization_results.items()
        }

    def save(
        self,
        result_path: Path,
        format_name: str | None = None,
        saving_options: SavingOptions = SAVING_OPTIONS_DEFAULT,
        *,
        allow_overwrite: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> list[str]:
        """Save the result to a file.

        Parameters
        ----------
        result_path : Path
            Path to the file where the result should be saved.
        format_name : str | None, optional
            Format in which to save the result, by default None.
        saving_options : SavingOptions, optional
            Options for saving the result, by default SAVING_OPTIONS_DEFAULT.
        allow_overwrite : bool, optional
            Whether to allow overwriting existing files, by default False.
        **kwargs : Any
            Additional keyword arguments for the specific plugin.

        Returns
        -------
        list[str]
            List of file paths where the result files were saved.
        """
        return save_result(
            self,
            result_path,
            format_name=format_name,
            saving_options=saving_options,
            allow_overwrite=allow_overwrite,
            **kwargs,
        )

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
            save_scheme(value, save_path, allow_overwrite=True)
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

    @staticmethod
    def extract_paths_from_serialization(
        result_file_path: Path, serialized: dict[str, Any]
    ) -> list[str]:
        """Extract file paths from serialized Result.

        Parameters
        ----------
        result_file_path : Path
            Path to the result file.
        serialized : dict[str, Any]
            Serialized representation of the Result.

        Yields
        ------
        list[str]
            List of file paths extracted from the serialization.
        """
        base_folder = result_file_path.parent
        project_paths_iterator = iter(
            (
                result_file_path,
                base_folder / serialized["scheme"],
                base_folder / serialized["initial_parameters"],
                base_folder / serialized["optimized_parameters"],
                base_folder / serialized["optimization_info"]["parameter_history"],
                base_folder / serialized["optimization_info"]["optimization_history"],
            )
        )
        result_iterators = tuple(
            OptimizationResult.extract_paths_from_serialization(
                base_folder / "optimization_results" / dataset_name, serialized
            )
            for dataset_name, serialized in serialized["optimization_results"].items()
        )
        return [
            relative_posix_path(path, base_path=Path())
            for path in chain(project_paths_iterator, *result_iterators)
        ]
