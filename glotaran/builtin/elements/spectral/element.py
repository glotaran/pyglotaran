from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal

import numpy as np

from glotaran.builtin.elements.spectral.shape import SpectralShape  # noqa: TCH001
from glotaran.model.data_model import DataModel
from glotaran.model.element import Element
from glotaran.model.element import ElementResult

if TYPE_CHECKING:
    import xarray as xr

    from glotaran.typing.types import ArrayLike


class SpectralDataModel(DataModel):
    spectral_axis_inverted: bool = False
    spectral_axis_scale: float = 1


class SpectralElement(Element):
    type: Literal["spectral"]  # type:ignore[assignment]
    register_as: ClassVar[str] = "spectral"
    dimension: str = "spectral"
    data_model_type: ClassVar[type[DataModel]] = SpectralDataModel  # type:ignore[valid-type]
    shapes: dict[str, SpectralShape.get_annotated_type()]  # type:ignore[valid-type]

    def calculate_matrix(  # type:ignore[override]
        self,
        model: SpectralDataModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
    ):
        compartments = list(self.shapes.keys())

        if model.spectral_axis_inverted:
            model_axis = model.spectral_axis_scale / model_axis
        elif model.spectral_axis_scale != 1:
            model_axis = model_axis * model.spectral_axis_scale

        dim1 = model_axis.size
        dim2 = len(self.shapes)
        matrix = np.zeros((dim1, dim2))

        for i, shape in enumerate(self.shapes.values()):
            matrix[:, i] += shape.calculate(model_axis)

        return compartments, matrix

    def add_to_result_data(  # type:ignore[override]
        self,
        model: SpectralDataModel,
        data: xr.Dataset,
        as_global: bool = False,
    ):
        if "spectrum" in data.coords:
            return

        elements = [m for m in model.elements if isinstance(m, SpectralElement)]
        shapes = [s for m in elements for s in m.shapes]

        data.coords["spectrum"] = shapes
        matrix = data.global_matrix if as_global else data.matrix
        clp_dim = "global_clp_label" if as_global else "clp_label"
        data["spectra"] = (
            (
                data.attrs["global_dimension"] if as_global else data.attrs["model_dimension"],
                "shape",
            ),
            matrix.sel({clp_dim: shapes}).to_numpy(),
        )

        if not hasattr(data, "global_matrix"):
            data["spectrum_associated_estimation"] = (
                (data.attrs["global_dimension"], "spectrum"),
                data.clp.sel(clp_label=shapes).data,
            )

    def create_result(
        self,
        model: SpectralDataModel,
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.Dataset,
        concentrations: xr.Dataset,
    ) -> ElementResult:
        shapes = list(self.shapes.keys())

        spectra_amplitude = amplitudes.sel(amplitude_label=shapes).rename(amplitude_label="shape")
        spectra_concentration = concentrations.sel(amplitude_label=shapes).rename(
            amplitude_label="shape"
        )

        return ElementResult(
            amplitudes={"spectrum": spectra_amplitude},
            concentrations={"spectrum": spectra_concentration},
        )
