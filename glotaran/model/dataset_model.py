"""The DatasetModel class."""

from __future__ import annotations

import contextlib
from collections import Counter
from typing import TYPE_CHECKING

import xarray as xr

from glotaran.model.item import model_item
from glotaran.model.item import model_item_validator

if TYPE_CHECKING:
    from typing import Any
    from typing import Generator

    from glotaran.model.megacomplex import Megacomplex
    from glotaran.model.model import Model
    from glotaran.parameter import Parameter


def create_dataset_model_type(properties: dict[str, Any]) -> type[DatasetModel]:
    """Create dataset model type for a model."""

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
    ) -> Generator[tuple[Parameter | str | None, Megacomplex | str], None, None]:
        """Iterates the dataset model's megacomplexes."""
        for i, megacomplex in enumerate(self.megacomplex):
            scale = self.megacomplex_scale[i] if self.megacomplex_scale is not None else None
            yield scale, megacomplex

    def iterate_global_megacomplexes(
        self,
    ) -> Generator[tuple[Parameter | str | None, Megacomplex | str], None, None]:
        """Iterates the dataset model's global megacomplexes."""
        for i, megacomplex in enumerate(self.global_megacomplex):
            scale = (
                self.global_megacomplex_scale[i]
                if self.global_megacomplex_scale is not None
                else None
            )
            yield scale, megacomplex

    def get_model_dimension(self) -> str:
        """Returns the dataset model's model dimension."""
        if len(self.megacomplex) == 0:
            raise ValueError(f"No megacomplex set for dataset model '{self.label}'")
        if isinstance(self.megacomplex[0], str):
            raise ValueError(f"Dataset model '{self.label}' was not filled")
        model_dimension = self.megacomplex[0].dimension
        if any(model_dimension != m.dimension for m in self.megacomplex):
            raise ValueError(
                f"Megacomplex dimensions do not match for dataset model '{self.label}'."
            )
        return model_dimension

    def finalize_data(self, dataset: xr.Dataset):
        """Finalize a dataset by applying all megacomplex finalize methods."""
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

    @model_item_validator(False)
    def ensure_unique_megacomplexes(self, model: Model) -> list[str]:
        """Ensure that unique megacomplexes are only used once per dataset.

        Parameters
        ----------
        model : Model
            Model object using this dataset model.

        Returns
        -------
        list[str]
            Error messages to be shown when the model gets validated.
        """
        errors = []

        def get_unique_errors(megacomplexes: list[str], is_global: bool) -> list[str]:
            unique_types = []
            for megacomplex_name in megacomplexes:
                with contextlib.suppress(KeyError):
                    megacomplex_instance = model.megacomplex[megacomplex_name]
                    if type(megacomplex_instance).glotaran_unique():
                        type_name = megacomplex_instance.type or megacomplex_instance.name
                        unique_types.append(type_name)
                this_errors = [
                    f"Multiple instances of unique{' global ' if is_global else ' '}"
                    f"megacomplex type {type_name!r} in dataset {self.label!r}"
                    for type_name, count in Counter(unique_types).most_common()
                    if count > 1
                ]

            return this_errors

        if self.megacomplex:
            errors += get_unique_errors(self.megacomplex, False)
        if self.global_megacomplex:
            errors += get_unique_errors(self.global_megacomplex, True)

        return errors

    @model_item_validator(False)
    def ensure_exclusive_megacomplexes(self, model: Model) -> list[str]:
        """Ensure that exclusive megacomplexes are the only megacomplex in the dataset model.

        Parameters
        ----------
        model : Model
            Model object using this dataset model.

        Returns
        -------
        list[str]
            Error messages to be shown when the model gets validated.
        """

        errors = []

        def get_exclusive_errors(megacomplexes: list[str]) -> list[str]:
            with contextlib.suppress(StopIteration):
                exclusive_megacomplex = next(
                    model.megacomplex[label]
                    for label in megacomplexes
                    if label in model.megacomplex
                    and type(model.megacomplex[label]).glotaran_exclusive()
                )
                if len(self.megacomplex) != 1:
                    return [
                        f"Megacomplex '{type(exclusive_megacomplex)}' is exclusive and cannot be "
                        f"combined with other megacomplex in dataset model '{self.label}'."
                    ]
            return []

        if self.megacomplex:
            errors += get_exclusive_errors(self.megacomplex)
        if self.global_megacomplex:
            errors += get_exclusive_errors(self.global_megacomplex)

        return errors
