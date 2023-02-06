from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from glotaran.builtin.megacomplexes.spectral.shape import SpectralShape
from glotaran.model import DataModel
from glotaran.model import Megacomplex


class SpectralDataModel(DataModel):
    spectral_axis_inverted: bool = False
    spectral_axis_scale: float = 1


class SpectralMegacomplex(Megacomplex):
    type: Literal["spectral"]
    dimension: str = "spectral"
    register_as = "spectral"
    data_model_type = SpectralDataModel
    shapes: dict[str, SpectralShape.get_annotated_type()]

    def calculate_matrix(
        self,
        model: SpectralDataModel,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
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

    def add_to_result_data(
        self,
        model: SpectralDataModel,
        data: xr.Dataset,
        as_global: bool = False,
    ):
        if "spectrum" in data.coords:
            return

        megacomplexes = [m for m in model.megacomplex if isinstance(m, SpectralMegacomplex)]
        shapes = [s for m in megacomplexes for s in m.shapes]

        data.coords["spectrum"] = shapes
        matrix = data.global_matrix if as_global else data.matrix
        clp_dim = "global_clp_label" if as_global else "clp_label"
        data["spectra"] = (
            data.attrs["global_dimension"] if as_global else data.attrs["model_dimension"],
            "shape",
        ), matrix.sel({clp_dim: shapes}).values

        if not hasattr(data, "global_matrix"):
            data["spectrum_associated_estimation"] = (
                (data.attrs["global_dimension"], "spectrum"),
                data.clp.sel(clp_label=shapes).data,
            )
