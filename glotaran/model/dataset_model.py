"""The DatasetModel class."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Generator

import xarray as xr

from glotaran.model.item import ItemIssue
from glotaran.model.item import ModelItem
from glotaran.model.item import ModelItemType
from glotaran.model.item import ParameterType
from glotaran.model.item import attribute
from glotaran.model.item import item
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.megacomplex import is_exclusive
from glotaran.model.megacomplex import is_unique

if TYPE_CHECKING:
    from glotaran.model.model import Model
    from glotaran.parameter import Parameter
    from glotaran.parameter import ParameterGroup


class ExclusiveMegacomplexIssue(ItemIssue):
    def __init__(self, label: str, megacomplex_type: str, is_global: bool):
        self._label = label
        self._type = megacomplex_type
        self._is_global = is_global

    def to_string(self) -> str:
        return (
            f"Exclusive {'global ' if self._is_global else ''}megacomplex '{self._label}' of "
            f"type '{self._type}' cannot be combined with other megacomplexes."
        )


class UniqueMegacomplexIssue(ItemIssue):
    def __init__(self, label: str, megacomplex_type: str, is_global: bool):
        self._label = label
        self._type = megacomplex_type
        self._is_global = is_global

    def to_string(self):
        return (
            f"Unique {'global ' if self._is_global else ''}megacomplex '{self._label}' of "
            f"type '{self._type}' can only be used once per dataset."
        )


def get_megacomplex_issues(
    value: list[str] | None, model: Model, is_global: bool
) -> list[ItemIssue]:
    issues = []

    if value is not None:
        labels = [v if isinstance(v, str) else v.label for v in value]
        megacomplexes = [model.megacomplex[label] for label in labels]
        for megacomplex in megacomplexes:
            megacomplex_type = megacomplex.__class__
            if is_exclusive(megacomplex_type) and len(megacomplexes) > 1:
                issues.append(
                    ExclusiveMegacomplexIssue(megacomplex.label, megacomplex.type, is_global)
                )
            if (
                is_unique(megacomplex_type)
                and len([m for m in megacomplexes if m.__class__ is megacomplex_type]) > 0
            ):
                issues.append(
                    UniqueMegacomplexIssue(megacomplex.label, megacomplex.type, is_global)
                )
    return issues


def validate_megacomplexes(
    value: list[str], model: Model, parameters: ParameterGroup | None
) -> list[ItemIssue]:
    return get_megacomplex_issues(value, model, False)


def validate_global_megacomplexes(
    value: list[str] | None, model: Model, parameters: ParameterGroup | None
) -> list[ItemIssue]:
    return get_megacomplex_issues(value, model, False)


@item
class DatasetModel(ModelItem):
    """A `DatasetModel` describes a dataset in terms of a glotaran model.
    It contains references to model items which describe the physical model for
    a given dataset.

    A general dataset descriptor assigns one or more megacomplexes and a scale
    parameter.
    """

    group: str = "default"
    force_index_dependent: bool = False
    megacomplex: list[ModelItemType[Megacomplex]] = attribute(validator=validate_megacomplexes)
    megacomplex_scale: list[ParameterType] | None = None
    global_megacomplex: list[ModelItemType[Megacomplex]] | None = attribute(
        alias="megacomplex", default=None, validator=validate_global_megacomplexes
    )
    global_megacomplex_scale: list[ParameterType] | None = None
    scale: ParameterType | None = None


def is_dataset_model_index_dependent(dataset_model: DatasetModel) -> bool:
    """Indicates if the dataset model is index dependent."""
    if dataset_model.force_index_dependent:
        return True
    return any(m.index_dependent(dataset_model) for m in dataset_model.megacomplex)


def has_dataset_model_global_model(dataset_model: DatasetModel) -> bool:
    """Indicates if the dataset model can model the global dimension."""
    return (
        dataset_model.global_megacomplex is not None and len(dataset_model.global_megacomplex) != 0
    )


def get_dataset_model_model_dimension(dataset_model: DatasetModel) -> str:
    """Returns the dataset model's model dimension."""
    if len(dataset_model.megacomplex) == 0:
        raise ValueError(f"No megacomplex set for dataset model '{dataset_model.label}'")
    if isinstance(dataset_model.megacomplex[0], str):
        raise ValueError(f"Dataset model '{dataset_model.label}' was not filled")
    model_dimension = dataset_model.megacomplex[0].dimension
    if any(model_dimension != m.dimension for m in dataset_model.megacomplex):
        raise ValueError(
            f"Megacomplex dimensions do not match for dataset model '{dataset_model.label}'."
        )
    return model_dimension


def iterate_dataset_model_megacomplexes(
    dataset_model: DatasetModel,
) -> Generator[tuple[Parameter | str | None, Megacomplex | str], None, None]:
    """Iterates the dataset model's megacomplexes."""
    for i, megacomplex in enumerate(dataset_model.megacomplex):
        scale = (
            dataset_model.megacomplex_scale[i]
            if dataset_model.megacomplex_scale is not None
            else None
        )
        yield scale, megacomplex


def iterate_dataset_model_global_megacomplexes(
    dataset_model: DatasetModel,
) -> Generator[tuple[Parameter | str | None, Megacomplex | str], None, None]:
    """Iterates the dataset model's global megacomplexes."""
    for i, megacomplex in enumerate(dataset_model.global_megacomplex):
        scale = (
            dataset_model.global_megacomplex_scale[i]
            if dataset_model.global_megacomplex_scale is not None
            else None
        )
        yield scale, megacomplex


def finalize_dataset_model(dataset_model: DatasetModel, dataset: xr.Dataset):
    """Finalize a dataset by applying all megacomplex finalize methods."""
    is_full_model = has_dataset_model_global_model(dataset_model)
    for megacomplex in dataset_model.megacomplex:
        megacomplex.finalize_data(dataset_model, dataset, is_full_model=is_full_model)
    if is_full_model:
        for megacomplex in dataset_model.global_megacomplex:
            megacomplex.finalize_data(
                dataset_model, dataset, is_full_model=is_full_model, as_global=True
            )
