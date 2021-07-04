"""The DatasetDescriptor class."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Generator
from typing import List

import xarray as xr

from glotaran.deprecation import deprecate
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

    A general dataset descriptor assigns one or more megacomplexes and a scale
    parameter.
    """

    def iterate_megacomplexes(self) -> Generator[tuple[Parameter | int, Megacomplex | str]]:
        for i, megacomplex in enumerate(self.megacomplex):
            scale = self.megacomplex_scale[i] if self.megacomplex_scale is not None else None
            yield scale, megacomplex

    def get_model_dimension(self) -> str:
        if not hasattr(self, "_model_dimension"):
            if len(self.megacomplex) == 0:
                raise ValueError(f"No megacomplex set for dataset descriptor '{self.label}'")
            if isinstance(self.megacomplex[0], str):
                raise ValueError(f"Dataset descriptor '{self.label}' was not filled")
            self._model_dimension = self.megacomplex[0].dimension
            if any(self._model_dimension != m.dimension for m in self.megacomplex):
                raise ValueError(
                    f"Megacomplex dimensions do not match for dataset descriptor '{self.label}'."
                )
        return self._model_dimension

    def overwrite_model_dimension(self, model_dimension: str):
        self._model_dimension = model_dimension

    # TODO: make explicit we only support 2 dimensions at present
    # TODO: the global dimension should become a flexible index (MultiIndex)
    # the user can then specify the name of the MultiIndex global dimension
    # using the function overwrite_global_dimension
    # e.g. in FLIM, x, y dimension may get 'flattened' to a MultiIndex 'pixel'
    def get_global_dimension(self) -> str:
        if not hasattr(self, "_global_dimension"):
            if not hasattr(self, "_data"):
                raise ValueError(f"Data not set for dataset descriptor '{self.label}'")
            self._global_dimension = [
                dim for dim in self._data.data.dims if dim != self.get_model_dimension()
            ][0]
        return self._global_dimension

    def overwrite_global_dimension(self, global_dimension: str):
        self._global_dimension = global_dimension

    def set_data(self, data: xr.Dataset) -> DatasetDescriptor:
        self._data = data
        return self

    def get_data(self) -> xr.Dataset:
        return self._data

    def index_dependent(self) -> bool:
        if hasattr(self, "_index_dependent"):
            return self._index_dependent
        return any(m.index_dependent(self) for m in self.megacomplex)

    def set_coords(self, coords: xr.Dataset):
        self._coords = coords

    def get_coords(self) -> xr.Dataset:
        if hasattr(self, "_coords"):
            return self._coords
        return self._data.coords

    @deprecate(
        deprecated_qual_name_usage=(
            "glotaran.model.dataset_descriptor.DatasetDescriptor.overwrite_index_dependent"
        ),
        new_qual_name_usage="",
        to_be_removed_in_version="0.6.0",
        importable_indices=(2, 2),
        has_glotaran_replacement=False,
    )
    def overwrite_index_dependent(self, index_dependent: bool):
        self._index_dependent = index_dependent
