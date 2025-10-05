from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from warnings import warn

import numpy as np
import xarray as xr
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SerializationInfo
from pydantic import ValidationInfo
from pydantic import computed_field
from pydantic import field_serializer
from pydantic import field_validator
from pydantic import model_validator

from glotaran.io import load_dataset
from glotaran.io import save_dataset
from glotaran.io.interface import SAVING_OPTIONS_DEFAULT
from glotaran.io.interface import SavingOptions
from glotaran.model.data_model import DataModel
from glotaran.model.data_model import iterate_data_model_elements
from glotaran.optimization.data import LinkedOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.estimation import OptimizationEstimation
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.optimization.penalty import calculate_clp_penalties
from glotaran.parameter.parameter import Parameter
from glotaran.plugin_system.base_registry import full_plugin_name
from glotaran.plugin_system.data_io_registration import get_data_io
from glotaran.utils.io import relative_posix_path
from glotaran.utils.pydantic_serde import context_is_dict
from glotaran.utils.pydantic_serde import save_folder_from_info

if TYPE_CHECKING:
    from collections.abc import Iterable

    from glotaran.model.element import Element
    from glotaran.model.experiment_model import ExperimentModel
    from glotaran.typing.types import ArrayLike


def add_svd_to_result_dataset(dataset: xr.Dataset, global_dim: str, model_dim: str) -> None:
    for name in ["data", "residual"]:
        if f"{name}_singular_values" in dataset:
            continue
        lsv, sv, rsv = np.linalg.svd(dataset[name].data, full_matrices=False)
        dataset[f"{name}_left_singular_vectors"] = (
            (model_dim, "left_singular_value_index"),
            lsv,
        )
        dataset[f"{name}_singular_values"] = (("singular_value_index"), sv)
        dataset[f"{name}_right_singular_vectors"] = (
            (global_dim, "right_singular_value_index"),
            rsv.T,
        )


class OptimizationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    elements: dict[str, xr.Dataset] = Field(default_factory=dict)
    activations: dict[str, xr.Dataset] = Field(default_factory=dict)
    input_data: xr.DataArray | xr.Dataset
    residuals: xr.DataArray | xr.Dataset | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fitted_data(self) -> xr.Dataset | xr.DataArray | None:
        """Fitted data derived from ``input_data - residuals``.

        Returns
        -------
        xr.Dataset | xr.DataArray | None
            Fitted data if ``residuals`` are set else ``None``
        """
        if self.residuals is None:
            warn(
                UserWarning("Residuals must be set to calculate fitted data."),
                stacklevel=2,
            )
            return None
        return self.input_data - self.residuals

    @field_serializer("input_data", "residuals", "fitted_data", when_used="json")
    def serialize_top_level_datasets(
        self, value: xr.DataArray | xr.Dataset | None, info: SerializationInfo
    ) -> str | tuple[str, str] | list[str] | None:
        """Save top level dataset to file and replace serialized value with the file path.

        This serialization is used when ``Model_dump`` is called in ``json`` mode.

        Parameters
        ----------
        value : xr.DataArray | xr.Dataset | None
            Value of the field to serialize.
        info : SerializationInfo
            Additional serialization information for the file including context passed to
            ``model_dump``.

        Returns
        -------
        str | tuple[str, str] | None
            ``str``: If dataset was saved with builtin plugin.
            ``tuple[str, str] | list[str]``: Only returned for ``input_data`` when it is filtered
                out and original dataset was loaded with a  3rd party plugin.
                The value has the shape: ``(file_path, io_plugin_name)``.
            ``None``: If value was not set (i.e. loaded from minimal save).

        Raises
        ------
        ValueError
            If serialization context does not contain the ``save_folder``.

        Examples
        --------
        >>> optimization_result.model_dump(mode="json", context={"save_folder": Path("result-1")})
        """
        if value is None:
            return None
        if context_is_dict(info) and (save_folder := save_folder_from_info(info)) is not None:
            saving_options: SavingOptions = info.context.get(
                "saving_options", SAVING_OPTIONS_DEFAULT
            )
            data_filter = saving_options.get("data_filter", set())
            data_format = saving_options.get("data_format", "nc")
            data_plugin = saving_options.get("data_plugin", None) or full_plugin_name(
                get_data_io(data_format)
            )
            if data_plugin.endswith(f"_{data_format}") is False:
                data_plugin = f"{data_plugin}_{data_format}"

            if info.field_name != "input_data" and info.field_name in data_filter:
                return None
            # Input data were loaded with ``load_dataset``
            if (
                info.field_name == "input_data"
                and info.field_name in data_filter
                and "source_path" in value.attrs
                and "io_plugin_name" in value.attrs
            ):
                original_io_plugin_name = value.attrs["io_plugin_name"]
                original_source_path = relative_posix_path(
                    value.attrs["source_path"], base_path=Path(save_folder)
                )
                # Only return the original source path if the plugin is from glotaran and
                # the plugin names match
                if original_io_plugin_name.startswith("glotaran.") and (
                    data_plugin
                    in {original_io_plugin_name, f"{original_io_plugin_name}_{data_format}"}
                ):
                    return original_source_path
                return (original_source_path, original_io_plugin_name)
            save_path = Path(save_folder) / f"{info.field_name}.{data_format}"
            save_dataset(
                value,
                save_path,
                format_name=data_plugin,
                allow_overwrite=True,
                update_source_path=info.field_name != "input_data",
            )
            return save_path.name

        msg = f"SerializationInfo context is missing 'save_folder':\n{info}"
        raise ValueError(msg)

    @field_validator("input_data", "residuals", mode="before")
    @classmethod
    def validate_top_level_datasets(
        cls,
        value: xr.DataArray | xr.Dataset | None | str | Path | tuple[str, str] | list[str],
        info: ValidationInfo,
    ) -> xr.DataArray | xr.Dataset | None:
        """Validate top level datasets and deserialize the if values are str or tuple[str, str].

        Parameters
        ----------
        value : xr.DataArray | xr.Dataset | None | str | Path | tuple[str, str]
            Value to validate.
        info : ValidationInfo
            Validation information for the field which may contain a context with ``save_folder``
            used to load data from file.

        Returns
        -------
        xr.DataArray | xr.Dataset | None

        Raises
        ------
        ValueError
            If the filed is ``input_data`` and the value is ``None``
        ValueError
            If the value is a tuple with a length unequal to two.
        ValueError
            If value isn't and expected type or can not be loaded from file due to missing
            ``save_folder`` in context.
        """
        if info.field_name == "input_data" and value is None:
            msg = "Input data cannot be None."
            raise ValueError(msg)
        if isinstance(value, (xr.Dataset, xr.DataArray)) or value is None:
            return value
        if context_is_dict(info) and (save_folder := save_folder_from_info(info)) is not None:
            saving_options: SavingOptions = info.context.get("saving_options", {})
            data_plugin = saving_options.get("data_plugin", None)

            if isinstance(value, (str, Path)):
                return load_dataset((Path(save_folder) / value).resolve(), format_name=data_plugin)
            if isinstance(value, list | tuple):
                if len(value) != 2 or not all(isinstance(v, str) for v in value):  # noqa: PLR2004
                    msg = (
                        f"Expected a tuple/list of relative file path and io plugin name for "
                        f"deserializing 'input_data' dataset, got: {value!r}"
                    )
                    raise ValueError(msg)
                rel_path, original_io_plugin_name = value
                return load_dataset(
                    (Path(save_folder) / rel_path).resolve(), format_name=original_io_plugin_name
                )
        msg = f"Unable to validate field {info.field_name}"
        raise ValueError(msg)

    @field_serializer("elements", "activations", when_used="json")
    def serialize_dataset_maps(
        self, value: dict[str, xr.DataArray | xr.Dataset], info: SerializationInfo
    ) -> dict[str, str] | None:
        """Save dataset collections to file and replace serialized item values with the file paths.

        This serialization is used when ``model_dump`` is called in ``json`` mode.


        Parameters
        ----------
        value : dict[str, xr.DataArray  |  xr.Dataset]
            Value of dataset collection to serialize.
        info : SerializationInfo
            Additional serialization information for the file including context passed to
            ``model_dump``.

        Returns
        -------
        dict[str, str] | None
            Mapping of entries and corresponding file paths.

        Raises
        ------
        ValueError
            If serialization context is missing ``save_folder``.

        Examples
        --------
        >>> optimization_result.model_dump(mode="json", context={"save_folder": Path("result-1")})
        """
        if context_is_dict(info) and (save_folder := save_folder_from_info(info)) is not None:
            saving_options: SavingOptions = info.context.get("saving_options", {})
            data_filter = saving_options.get("data_filter", set())
            data_format = saving_options.get("data_format", "nc")
            if info.field_name in data_filter:
                return {}
            serialization_mapping = {}
            for key, dataset in value.items():
                save_path = Path(save_folder) / info.field_name / f"{key}.{data_format}"
                save_dataset(dataset, save_path, allow_overwrite=True)
                serialization_mapping[key] = save_path.name
            return serialization_mapping
        msg = f"SerializationInfo context is missing 'save_folder':\n{info}"
        raise ValueError(msg)

    @field_validator("elements", "activations", mode="before")
    @classmethod
    def validate_dataset_maps(
        cls,
        value: dict[str, str | Path] | dict[str, xr.DataArray | xr.Dataset],
        info: ValidationInfo,
    ) -> dict[str, xr.DataArray | xr.Dataset]:
        """Validate dataset maps and deserialize them if item values are ``str`` or ``Path``.

        Parameters
        ----------
        value : dict[str, str | Path] | dict[str, xr.DataArray  |  xr.Dataset]
            _description_
        info : ValidationInfo
            Validation information for the field which may contain a context with ``save_folder``
            used to load data from file.

        Returns
        -------
        dict[str, xr.DataArray | xr.Dataset]
        """
        if (
            context_is_dict(info)
            and (save_folder := save_folder_from_info(info)) is not None
            and all(isinstance(v, (str, Path)) for v in value.values())
        ):
            saving_options: SavingOptions = info.context.get(
                "saving_options", SAVING_OPTIONS_DEFAULT
            )
            data_plugin = saving_options.get("data_plugin", None)
            return {
                item_label: load_dataset(
                    (Path(save_folder) / info.field_name / item_value).resolve(),
                    format_name=data_plugin,
                )
                for item_label, item_value in value.items()
                if isinstance(item_value, (str, Path))
            }
        return value

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, value: Any) -> Any:  # noqa: ANN401
        """Remove computed field before attempting to deserialize model."""
        if isinstance(value, dict):
            value.pop("fitted_data", None)
            return value
        return value


@dataclass
class OptimizationObjectiveResult:
    optimization_results: dict[str, OptimizationResult]
    additional_penalty: float
    clp_size: int


class OptimizationObjective:
    def __init__(self, model: ExperimentModel) -> None:
        self._data = (
            LinkedOptimizationData.from_experiment_model(model)
            if len(model.datasets) > 1
            else OptimizationData(next(iter(model.datasets.values())))
        )
        self._model = model

    def calculate_matrices(self) -> list[OptimizationMatrix]:
        if isinstance(self._data, OptimizationData):
            return OptimizationMatrix.from_data(self._data).as_global_list(self._data.global_axis)
        return OptimizationMatrix.from_linked_data(self._data)

    def calculate_reduced_matrices(
        self, matrices: list[OptimizationMatrix]
    ) -> list[OptimizationMatrix]:
        return [
            matrices[i].reduce(index, self._model.clp_relations, copy=True)
            for i, index in enumerate(self._data.global_axis)
        ]

    def calculate_estimations(
        self, reduced_matrices: list[OptimizationMatrix]
    ) -> list[OptimizationEstimation]:
        return [
            OptimizationEstimation.calculate(matrix.array, data, self._model.residual_function)
            for matrix, data in zip(reduced_matrices, self._data.data_slices, strict=True)
        ]

    def resolve_estimations(
        self,
        matrices: list[OptimizationMatrix],
        reduced_matrices: list[OptimizationMatrix],
        estimations: list[OptimizationEstimation],
    ) -> list[OptimizationEstimation]:
        return [
            e.resolve_clp(m.clp_axis, r.clp_axis, i, self._model.clp_relations)
            for e, m, r, i in zip(
                estimations, matrices, reduced_matrices, self._data.global_axis, strict=True
            )
        ]

    def calculate_global_penalty(self) -> ArrayLike:
        assert isinstance(self._data, OptimizationData)
        assert self._data.flat_data is not None
        _, _, matrix = OptimizationMatrix.from_global_data(self._data)
        return OptimizationEstimation.calculate(
            matrix.array,
            self._data.flat_data,
            self._model.residual_function,
        ).residual

    def calculate(self) -> ArrayLike:
        if isinstance(self._data, OptimizationData) and self._data.is_global:
            return self.calculate_global_penalty()
        matrices = self.calculate_matrices()
        reduced_matrices = self.calculate_reduced_matrices(matrices)
        estimations = self.calculate_estimations(reduced_matrices)

        penalties = [e.residual for e in estimations]
        if len(self._model.clp_penalties) > 0:
            estimations = self.resolve_estimations(matrices, reduced_matrices, estimations)
            penalties.append(
                calculate_clp_penalties(
                    matrices,
                    estimations,
                    self._data.global_axis,
                    self._model.clp_penalties,
                )
            )
        return np.concatenate(penalties)

    def get_global_indices(self, label: str) -> list[int]:
        assert isinstance(self._data, LinkedOptimizationData)
        return [
            i
            for i, group_label in enumerate(self._data.group_labels)
            if label in self._data.group_definitions[group_label]
        ]

    def create_result_dataset(self, label: str, data: OptimizationData) -> xr.Dataset:
        assert isinstance(data.model.data, xr.Dataset)
        dataset = data.model.data.copy()
        if dataset.data.dims != (data.model_dimension, data.global_dimension):
            dataset["data"] = dataset.data.T
        dataset["data"].attrs = data.original_dataset_attributes.copy()
        dataset.attrs["model_dimension"] = data.model_dimension
        dataset.attrs["global_dimension"] = data.global_dimension
        dataset.coords[data.model_dimension] = data.model_axis
        dataset.coords[data.global_dimension] = data.global_axis
        if isinstance(self._data, LinkedOptimizationData):
            scale = self._data.scales[label]
            dataset.attrs["scale"] = scale.value if isinstance(scale, Parameter) else scale
        return dataset

    def create_global_result(self) -> OptimizationObjectiveResult:
        label = next(iter(self._model.datasets.keys()))
        assert isinstance(self._data, OptimizationData)
        result_dataset = self.create_result_dataset(label, self._data)

        global_dim = result_dataset.attrs["global_dimension"]
        global_axis = result_dataset.coords[global_dim]
        model_dim = result_dataset.attrs["model_dimension"]
        model_axis = result_dataset.coords[model_dim]

        matrix = OptimizationMatrix.from_data(self._data).to_data_array(
            global_dim, global_axis.to_numpy(), model_dim, model_axis.to_numpy()
        )
        global_matrix = OptimizationMatrix.from_data(self._data, global_matrix=True).to_data_array(
            model_dim, model_axis.to_numpy(), global_dim, global_axis.to_numpy()
        )
        _, _, full_matrix = OptimizationMatrix.from_global_data(self._data)

        assert self._data.flat_data is not None
        estimation = OptimizationEstimation.calculate(
            full_matrix.array,
            self._data.flat_data,
            self._data.model.residual_function,
        )
        clp = xr.DataArray(
            estimation.clp.reshape(
                (len(global_matrix.amplitude_label), len(matrix.amplitude_label))
            ),
            coords={
                "global_clp_label": global_matrix.amplitude_label.to_numpy(),
                "clp_label": matrix.amplitude_label.to_numpy(),
            },
            dims=["global_clp_label", "clp_label"],
        )
        result_dataset["residual"] = xr.DataArray(
            estimation.residual.reshape(global_axis.size, model_axis.size),
            coords=((global_dim, global_axis.to_numpy()), (model_dim, model_axis.to_numpy())),
        ).T
        result_dataset.attrs["root_mean_square_error"] = np.sqrt(
            (result_dataset.residual.to_numpy() ** 2).sum() / sum(result_dataset.residual.shape)
        )
        clp_size = len(matrix.amplitude_label) + len(global_matrix.amplitude_label)
        self._data.unweight_result_dataset(result_dataset)

        add_svd_to_result_dataset(result_dataset, global_dim, model_dim)
        result = OptimizationResult(
            input_data=result_dataset.data,
            residuals=result_dataset.residual,
            elements={
                label: xr.Dataset(
                    {
                        "amplitudes": clp,
                        "global_concentrations": global_matrix,
                        "model_concentrations": matrix,
                    }
                )
            },
        )
        return OptimizationObjectiveResult(
            optimization_results={label: result}, clp_size=clp_size, additional_penalty=0
        )

    def create_single_dataset_result(self) -> OptimizationObjectiveResult:
        assert isinstance(self._data, OptimizationData)
        if self._data.is_global:
            return self.create_global_result()

        label = next(iter(self._model.datasets.keys()))
        result_dataset = self.create_result_dataset(label, self._data)

        global_dim = result_dataset.attrs["global_dimension"]
        global_axis = result_dataset.coords[global_dim]
        model_dim = result_dataset.attrs["model_dimension"]
        model_axis = result_dataset.coords[model_dim]

        concentrations = OptimizationMatrix.from_data(self._data)
        additional_penalty = 0

        clp_concentration = self.calculate_reduced_matrices(
            concentrations.as_global_list(self._data.global_axis)
        )
        clp_size = sum(len(c.clp_axis) for c in clp_concentration)
        estimations = self.resolve_estimations(
            concentrations.as_global_list(self._data.global_axis),
            clp_concentration,
            self.calculate_estimations(clp_concentration),
        )
        amplitude_coords = {
            global_dim: global_axis,
            "amplitude_label": concentrations.clp_axis,
        }
        amplitudes = xr.DataArray(
            [e.clp for e in estimations], dims=amplitude_coords.keys(), coords=amplitude_coords
        )
        concentration = concentrations.to_data_array(
            global_dim, global_axis, model_dim, model_axis
        )

        residual_dims = (global_dim, model_dim)
        result_dataset["residual"] = xr.DataArray(
            [e.residual for e in estimations], dims=residual_dims
        ).T
        result_dataset.attrs["root_mean_square_error"] = np.sqrt(
            (result_dataset.residual.to_numpy() ** 2).sum() / sum(result_dataset.residual.shape)
        )
        additional_penalty = sum(
            calculate_clp_penalties(
                [concentrations],
                estimations,
                global_axis,
                self._model.clp_penalties,
            )
        )
        element_results = self.create_element_results(
            self._model.datasets[label], global_dim, model_dim, amplitudes, concentration
        )
        activations = self.create_data_model_results(
            label, global_dim, model_dim, amplitudes, concentration
        )

        self._data.unweight_result_dataset(result_dataset)
        add_svd_to_result_dataset(result_dataset, global_dim, model_dim)
        input_data = result_dataset.data
        input_data.attrs |= self._data.original_dataset_attributes.copy()
        result = OptimizationResult(
            input_data=input_data,
            residuals=result_dataset.residual,
            elements=element_results,
            activations=activations,
        )
        return OptimizationObjectiveResult(
            optimization_results={label: result},
            additional_penalty=additional_penalty,
            clp_size=clp_size,
        )

    def create_multi_dataset_result(self) -> OptimizationObjectiveResult:
        assert isinstance(self._data, LinkedOptimizationData)
        dataset_concentrations = {
            label: OptimizationMatrix.from_data(data) for label, data in self._data.data.items()
        }
        full_concentration = OptimizationMatrix.from_linked_data(
            self._data, dataset_concentrations
        )
        estimated_amplitude_axes = [concentration.clp_axis for concentration in full_concentration]
        clp_concentration = self.calculate_reduced_matrices(full_concentration)
        clp_size = sum(len(concentration.clp_axis) for concentration in clp_concentration)

        estimations = self.resolve_estimations(
            full_concentration,
            clp_concentration,
            self.calculate_estimations(clp_concentration),
        )
        additional_penalty = sum(
            calculate_clp_penalties(
                full_concentration,
                estimations,
                self._data.global_axis,
                self._model.clp_penalties,
            )
        )

        results = {
            label: self.create_dataset_result(
                label,
                data,
                dataset_concentrations[label],
                estimated_amplitude_axes,
                estimations,
            )
            for label, data in self._data.data.items()
        }
        return OptimizationObjectiveResult(
            optimization_results=results,
            clp_size=clp_size,
            additional_penalty=additional_penalty,
        )

    def get_dataset_amplitudes(
        self,
        label: str,
        estimated_amplitude_axes: list[list[str]],
        estimated_amplitudes: list[OptimizationEstimation],
        amplitude_axis: ArrayLike,
        global_dim: str,
        global_axis: ArrayLike,
    ) -> xr.DataArray:
        assert isinstance(self._data, LinkedOptimizationData)

        global_indices = self.get_global_indices(label)
        coords = {
            global_dim: global_axis,
            "amplitude_label": amplitude_axis,
        }
        return xr.DataArray(
            [
                [
                    estimated_amplitudes[i].clp[estimated_amplitude_axes[i].index(amplitude_label)]
                    for amplitude_label in amplitude_axis
                ]
                for i in global_indices
            ],
            dims=coords.keys(),
            coords=coords,
        )

    def get_dataset_residual(
        self,
        label: str,
        estimations: list[OptimizationEstimation],
        model_dim: str,
        model_axis: ArrayLike,
        global_dim: str,
        global_axis: ArrayLike,
    ) -> xr.DataArray:
        assert isinstance(self._data, LinkedOptimizationData)

        global_indices = self.get_global_indices(label)
        coords = {global_dim: global_axis, model_dim: model_axis}
        offsets = []
        for i in global_indices:
            group_label = self._data._group_labels[i]  # noqa: SLF001
            group_index = self._data.group_definitions[group_label].index(label)
            offsets.append(sum(self._data.group_sizes[group_label][:group_index]))
        size = model_axis.size
        return xr.DataArray(
            [
                estimations[i].residual[offset : offset + size]
                for i, offset in zip(global_indices, offsets, strict=True)
            ],
            dims=coords.keys(),
            coords=coords,
        ).T

    def create_element_results(
        self,
        model: DataModel,
        global_dim: str,
        model_dim: str,
        amplitudes: xr.DataArray,
        concentrations: xr.DataArray,
    ) -> dict[str, xr.Dataset]:
        assert any(isinstance(element, str) for element in model.elements) is False
        return {
            element.label: element.create_result_with_uid(
                model, global_dim, model_dim, amplitudes, concentrations
            )
            for element in cast("list[Element]", model.elements)
        }

    def create_dataset_result(
        self,
        label: str,
        data: OptimizationData,
        concentration: OptimizationMatrix,
        estimated_amplitude_axes: list[list[str]],
        estimations: list[OptimizationEstimation],
    ) -> OptimizationResult:
        assert isinstance(self._data, LinkedOptimizationData)
        result_dataset = self.create_result_dataset(label, data)

        global_dim = result_dataset.attrs["global_dimension"]
        global_axis = result_dataset.coords[global_dim]
        model_dim = result_dataset.attrs["model_dimension"]
        model_axis = result_dataset.coords[model_dim]

        result_dataset["residual"] = self.get_dataset_residual(
            label, estimations, model_dim, model_axis, global_dim, global_axis
        )
        result_dataset.attrs["root_mean_square_error"] = np.sqrt(
            (result_dataset.residual.to_numpy() ** 2).sum() / sum(result_dataset.residual.shape)
        )
        self._data.data[label].unweight_result_dataset(result_dataset)
        result_dataset["fit"] = result_dataset.data - result_dataset.residual
        add_svd_to_result_dataset(result_dataset, global_dim, model_dim)

        concentrations = concentration.to_data_array(
            global_dim, global_axis, model_dim, model_axis
        )
        amplitudes = self.get_dataset_amplitudes(
            label,
            estimated_amplitude_axes,
            estimations,
            concentrations.amplitude_label,
            global_dim,
            global_axis,
        )
        element_results = self.create_element_results(
            self._model.datasets[label], global_dim, model_dim, amplitudes, concentrations
        )
        activations = self.create_data_model_results(
            label, global_dim, model_dim, amplitudes, concentrations
        )

        return OptimizationResult(
            input_data=result_dataset.data,
            residuals=result_dataset.residual,
            elements=element_results,
            activations=activations,
        )

    def create_data_model_results(
        self,
        label: str,
        global_dim: str,
        model_dim: str,
        amplitudes: xr.DataArray,
        concentrations: xr.DataArray,
    ) -> dict[str, xr.Dataset]:
        result: dict[str, xr.Dataset] = {}
        data_model = self._model.datasets[label]
        assert any(isinstance(e, str) for _, e in iterate_data_model_elements(data_model)) is False
        for data_model_cls in {
            e.__class__.data_model_type
            for _, e in cast(
                "Iterable[tuple[Any, Element]]", iterate_data_model_elements(data_model)
            )
            if e.__class__.data_model_type is not None
        }:
            result = result | cast("type[DataModel]", data_model_cls).create_result(
                data_model,
                global_dim,
                model_dim,
                amplitudes,
                concentrations,
            )
        return result

    def get_result(self) -> OptimizationObjectiveResult:
        return (
            self.create_single_dataset_result()
            if isinstance(self._data, OptimizationData)
            else self.create_multi_dataset_result()
        )
