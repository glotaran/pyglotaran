"""The DatasetModel class."""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.model.item import model_item
from glotaran.model.item import model_item_validator

if TYPE_CHECKING:
    from typing import Any
    from typing import Generator
    from typing import Hashable

    from glotaran.model.megacomplex import Megacomplex
    from glotaran.model.model import Model
    from glotaran.parameter import Parameter


def create_dataset_model_type(properties: dict[str, Any]) -> type[DatasetModel]:
    @model_item(properties=properties)
    class ModelDatasetModel(DatasetModel):
        pass

    return ModelDatasetModel


class DatasetModel:
    """A `DatasetModel` describes a dataset in terms of a glotaran model.
    It contains references to model items which describe the physical model for
    a given dataset.

    A general dataset descriptor assigns one or more megacomplexes and a scale
    parameter.
    """

    def iterate_megacomplexes(
        self,
    ) -> Generator[tuple[Parameter | int | None, Megacomplex | str], None, None]:
        """Iterates of der dataset model's megacomplexes."""
        for i, megacomplex in enumerate(self.megacomplex):
            scale = self.megacomplex_scale[i] if self.megacomplex_scale is not None else None
            yield scale, megacomplex

    def iterate_global_megacomplexes(
        self,
    ) -> Generator[tuple[Parameter | int | None, Megacomplex | str], None, None]:
        """Iterates of der dataset model's global megacomplexes."""
        for i, megacomplex in enumerate(self.global_megacomplex):
            scale = (
                self.global_megacomplex_scale[i]
                if self.global_megacomplex_scale is not None
                else None
            )
            yield scale, megacomplex

    def get_model_dimension(self) -> str:
        """Returns the dataset model's model dimension."""
        if not hasattr(self, "_model_dimension"):
            if len(self.megacomplex) == 0:
                raise ValueError(f"No megacomplex set for dataset model '{self.label}'")
            if isinstance(self.megacomplex[0], str):
                raise ValueError(f"Dataset model '{self.label}' was not filled")
            self._model_dimension = self.megacomplex[0].dimension
            if any(self._model_dimension != m.dimension for m in self.megacomplex):
                raise ValueError(
                    f"Megacomplex dimensions do not match for dataset model '{self.label}'."
                )
        return self._model_dimension

    def finalize_data(self, dataset: xr.Dataset) -> None:
        is_full_model = self.has_global_model()
        for megacomplex in self.megacomplex:
            megacomplex.finalize_data(self, dataset, is_full_model=is_full_model)
        if is_full_model:
            for megacomplex in self.global_megacomplex:
                megacomplex.finalize_data(
                    self, dataset, is_full_model=is_full_model, as_global=True
                )

    def overwrite_model_dimension(self, model_dimension: str) -> None:
        """Overwrites the dataset model's model dimension."""
        self._model_dimension = model_dimension

    # TODO: make explicit we only support 2 dimensions at present
    # TODO: the global dimension should become a flexible index (MultiIndex)
    # the user can then specify the name of the MultiIndex global dimension
    # using the function overwrite_global_dimension
    # e.g. in FLIM, x, y dimension may get 'flattened' to a MultiIndex 'pixel'
    def get_global_dimension(self) -> str:
        """Returns the dataset model's global dimension."""
        if not hasattr(self, "_global_dimension"):
            if self.has_global_model():
                if isinstance(self.global_megacomplex[0], str):
                    raise ValueError(f"Dataset model '{self.label}' was not filled")
                self._global_dimension = self.global_megacomplex[0].dimension
                if any(self._global_dimension != m.dimension for m in self.global_megacomplex):
                    raise ValueError(
                        "Global megacomplex dimensions do not "
                        f"match for dataset model '{self.label}'."
                    )
            elif hasattr(self, "_coords"):
                return next(dim for dim in self._coords if dim != self.get_model_dimension())
            else:
                if not hasattr(self, "_data"):
                    raise ValueError(f"Data not set for dataset model '{self.label}'")
                self._global_dimension = next(
                    dim for dim in self._data.data.dims if dim != self.get_model_dimension()
                )
        return self._global_dimension

    def overwrite_global_dimension(self, global_dimension: str) -> None:
        """Overwrites the dataset model's global dimension."""
        self._global_dimension = global_dimension

    def swap_dimensions(self) -> None:
        """Swaps the dataset model's global and model dimension."""
        global_dimension = self.get_model_dimension()
        model_dimension = self.get_global_dimension()
        self.overwrite_global_dimension(global_dimension)
        self.overwrite_model_dimension(model_dimension)

    def set_data(self, dataset: xr.Dataset) -> DatasetModel:
        """Sets the dataset model's data."""
        self._coords = {name: dim.values for name, dim in dataset.coords.items()}
        self._data: np.ndarray = dataset.data.values
        self._weight: np.ndarray | None = dataset.weight.values if "weight" in dataset else None
        if self._weight is not None:
            self._data = self._data * self._weight
        return self

    def get_data(self) -> np.ndarray:
        """Gets the dataset model's data."""
        return self._data

    def get_weight(self) -> np.ndarray | None:
        """Gets the dataset model's weight."""
        return self._weight

    def is_index_dependent(self) -> bool:
        """Indicates if the dataset model is index dependent."""
        if hasattr(self, "_index_dependent"):
            return self._index_dependent
        return any(m.index_dependent(self) for m in self.megacomplex)

    def overwrite_index_dependent(self, index_dependent: bool):
        """Overrides the index dependency of the dataset"""
        self._index_dependent = index_dependent

    def has_global_model(self) -> bool:
        """Indicates if the dataset model can model the global dimension."""
        return self.global_megacomplex is not None and len(self.global_megacomplex) != 0

    def set_coordinates(self, coords: dict[str, np.ndarray]):
        """Sets the dataset model's coordinates."""
        self._coords = coords

    def get_coordinates(self) -> dict[Hashable, np.ndarray]:
        """Gets the dataset model's coordinates."""
        return self._coords

    def get_model_axis(self) -> np.ndarray:
        """Gets the dataset model's model axis."""
        return self._coords[self.get_model_dimension()]

    def get_global_axis(self) -> np.ndarray:
        """Gets the dataset model's global axis."""
        return self._coords[self.get_global_dimension()]

    @model_item_validator(False)
    def ensure_unique_megacomplexes(self, model: Model) -> list[str]:
        """Ensure that unique megacomplexes Are only used once per dataset.

        Parameters
        ----------
        model : Model
            Model object using this dataset model.

        Returns
        -------
        list[str]
            Error messages to be shown when the model gets validated.
        """
        glotaran_unique_megacomplex_types = []

        for megacomplex_name in self.megacomplex:
            try:
                megacomplex_instance = model.megacomplex[megacomplex_name]
                if type(megacomplex_instance).glotaran_unique() is True:
                    type_name = megacomplex_instance.type or megacomplex_instance.name
                    glotaran_unique_megacomplex_types.append(type_name)
            except KeyError:
                pass

        return [
            f"Multiple instances of unique megacomplex type {type_name!r} "
            f"in dataset {self.label!r}"
            for type_name, count in Counter(glotaran_unique_megacomplex_types).most_common()
            if count > 1
        ]
