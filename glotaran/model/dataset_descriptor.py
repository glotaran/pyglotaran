"""The DatasetDescriptor class."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Generator
from typing import List

import xarray as xr

from glotaran.model.attribute import model_attribute
from glotaran.parameter import Parameter

if TYPE_CHECKING:
    from glotaran.model.megacomplex import Megacomplex


@model_attribute(
    properties={
        "megacomplex": List[str],
        "megacomplex_scale": {"type": List[Parameter], "default": None, "allow_none": True},
        "scale": {"type": Parameter, "default": None, "allow_none": True},
    }
)
class DatasetDescriptor:
    """A `DatasetDescriptor` describes a dataset in terms of a glotaran model.
    It contains references to model items which describe the physical model for
    a given dataset.

    A general dataset describtor assigns one or more megacomplexes and a scale
    parameter.
    """

    def iterate_megacomplexes(self) -> Generator[tuple[Parameter | int, Megacomplex | str]]:
        for i, megacomplex in enumerate(self.megacomplex):
            scale = self.megacomplex_scale[i] if self.megacomplex_scale is not None else None
            yield scale, megacomplex

    def get_model_dimension(self) -> str:
        if not hasattr(self, "_model_dimension"):
            if len(self.megacomplex) == 0:
                raise ValueError(f"No megacomplex set for dataset describtor '{self.label}'")
            if isinstance(self.megacomplex[0], str):
                raise ValueError(f"Dataset describtor '{self.label}' was not filled")
            self._model_dimension = self.megacomplex[0].dimension
            if any(self._model_dimension != m.dimension for m in self.megacomplex):
                raise ValueError(
                    f"Megacomplex dimensions do not match for dataset describtor '{self.label}'."
                )
        return self._model_dimension

    def get_global_dimension(self) -> str:
        if not hasattr(self, "_data"):
            raise ValueError(f"Data not set for dataset describtor '{self.label}'")
        return [dim for dim in self._data.data.dims if dim != self.get_model_dimension()][0]

    def set_data(self, data: xr.Dataset):
        self._data = data
