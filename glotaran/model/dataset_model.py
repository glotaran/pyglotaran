"""This module contains the dataset model."""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

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
    from glotaran.parameter import Parameters


class ExclusiveMegacomplexIssue(ItemIssue):
    """Issue for exclusive megacomplexes."""

    def __init__(self, label: str, megacomplex_type: str, is_global: bool):
        """Create an ExclusiveMegacomplexIssue.

        Parameters
        ----------
        label : str
            The megacomplex label.
        megacomplex_type : str
            The megacomplex type.
        is_global : bool
            Whether the megacomplex is global.
        """
        self._label = label
        self._type = megacomplex_type
        self._is_global = is_global

    def to_string(self) -> str:
        """Get the issue as string.

        Returns
        -------
        str
        """
        return (
            f"Exclusive {'global ' if self._is_global else ''}megacomplex '{self._label}' of "
            f"type '{self._type}' cannot be combined with other megacomplexes."
        )


class UniqueMegacomplexIssue(ItemIssue):
    """Issue for unique megacomplexes."""

    def __init__(self, label: str, megacomplex_type: str, is_global: bool):
        """Create a UniqueMegacomplexIssue.

        Parameters
        ----------
        label : str
            The megacomplex label.
        megacomplex_type : str
            The megacomplex type.
        is_global : bool
            Whether the megacomplex is global.
        """
        self._label = label
        self._type = megacomplex_type
        self._is_global = is_global

    def to_string(self):
        """Get the issue as string.

        Returns
        -------
        str
        """
        return (
            f"Unique {'global ' if self._is_global else ''}megacomplex '{self._label}' of "
            f"type '{self._type}' can only be used once per dataset."
        )


def get_megacomplex_issues(
    value: list[str | Megacomplex] | None, model: Model, is_global: bool
) -> list[ItemIssue]:
    """Get issues for megacomplexes.

    Parameters
    ----------
    value: list[str | Megacomplex] | None
        A list of megacomplexes.
    model: Model
        The model.
    is_global: bool
        Whether the megacomplexes are global.

    Returns
    -------
    list[ItemIssue]
    """
    issues: list[ItemIssue] = []

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
                and len([m for m in megacomplexes if m.__class__ is megacomplex_type]) > 1
            ):
                issues.append(
                    UniqueMegacomplexIssue(megacomplex.label, megacomplex.type, is_global)
                )
    return issues


def validate_megacomplexes(
    value: list[str | Megacomplex],
    dataset_model: DatasetModel,
    model: Model,
    parameters: Parameters | None,
) -> list[ItemIssue]:
    """Get issues for dataset model megacomplexes.

    Parameters
    ----------
    value: list[str | Megacomplex]
        A list of megacomplexes.
    dataset_model: DatasetModel
        The dataset model.
    model: Model
        The model.
    parameters: Parameters | None,
        The parameters.

    Returns
    -------
    list[ItemIssue]
    """
    return get_megacomplex_issues(value, model, False)


def validate_global_megacomplexes(
    value: list[str | Megacomplex] | None,
    dataset_model: DatasetModel,
    model: Model,
    parameters: Parameters | None,
) -> list[ItemIssue]:
    """Get issues for dataset model global megacomplexes.

    Parameters
    ----------
    value: list[str | Megacomplex] | None
        A list of megacomplexes.
    dataset_model: DatasetModel
        The dataset model.
    model: Model
        The model.
    parameters: Parameters | None,
        The parameters.

    Returns
    -------
    list[ItemIssue]
    """
    return get_megacomplex_issues(value, model, False)


@item
class DatasetModel(ModelItem):
    """A model for datasets."""

    group: str = "default"
    force_index_dependent: bool = False
    megacomplex: list[ModelItemType[Megacomplex]] = attribute(
        validator=validate_megacomplexes  # type:ignore[arg-type]
    )
    megacomplex_scale: list[ParameterType] | None = None
    global_megacomplex: list[ModelItemType[Megacomplex]] | None = attribute(
        alias="megacomplex",
        default=None,
        validator=validate_global_megacomplexes,  # type:ignore[arg-type]
    )
    global_megacomplex_scale: list[ParameterType] | None = None
    scale: ParameterType | None = None


def has_dataset_model_global_model(dataset_model: DatasetModel) -> bool:
    """Check if the dataset model can model the global dimension.

    Parameters
    ----------
    dataset_model: DatasetModel
        The dataset model.

    Returns
    -------
    bool
    """
    return (
        dataset_model.global_megacomplex is not None and len(dataset_model.global_megacomplex) != 0
    )


def get_dataset_model_model_dimension(dataset_model: DatasetModel) -> str:
    """Get the dataset model's model dimension.

    Parameters
    ----------
    dataset_model: DatasetModel
        The dataset model.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        Raised if the dataset model does not have megacomplexes or if it is not filled.
    """
    if len(dataset_model.megacomplex) == 0:
        raise ValueError(f"No megacomplex set for dataset model '{dataset_model.label}'.")
    if any(isinstance(m, str) for m in dataset_model.megacomplex):
        raise ValueError(f"Dataset model '{dataset_model.label}' was not filled.")
    model_dimension: str = dataset_model.megacomplex[
        0
    ].dimension  # type:ignore[union-attr, assignment]
    if any(
        model_dimension != m.dimension  # type:ignore[union-attr]
        for m in dataset_model.megacomplex
    ):
        raise ValueError(
            f"Megacomplex dimensions do not match for dataset model '{dataset_model.label}'."
        )
    return model_dimension


def iterate_dataset_model_megacomplexes(
    dataset_model: DatasetModel,
) -> Generator[tuple[Parameter | str | None, Megacomplex | str], None, None]:
    """Iterate the dataset model's megacomplexes.

    Parameters
    ----------
    dataset_model: DatasetModel
        The dataset model.

    Yields
    ------
    tuple[Parameter | str | None, Megacomplex | str]
        A scale and megacomplex.
    """
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
    """Iterate the dataset model's global megacomplexes.

    Parameters
    ----------
    dataset_model: DatasetModel
        The dataset model.

    Yields
    ------
    tuple[Parameter | str | None, Megacomplex | str]
        A scale and megacomplex.
    """
    if dataset_model.global_megacomplex is None:
        return
    for i, megacomplex in enumerate(dataset_model.global_megacomplex):
        scale = (
            dataset_model.global_megacomplex_scale[i]
            if dataset_model.global_megacomplex_scale is not None
            else None
        )
        yield scale, megacomplex


def finalize_dataset_model(dataset_model: DatasetModel, dataset: xr.Dataset):
    """Finalize a dataset by applying all megacomplex finalize methods.

    Parameters
    ----------
    dataset_model: DatasetModel
        The dataset model.
    dataset: xr.Dataset
        The dataset.
    """
    is_full_model = has_dataset_model_global_model(dataset_model)
    for megacomplex in dataset_model.megacomplex:
        megacomplex.finalize_data(  # type:ignore[union-attr]
            dataset_model, dataset, is_full_model=is_full_model
        )
    if is_full_model and dataset_model.global_megacomplex is not None:
        for megacomplex in dataset_model.global_megacomplex:
            megacomplex.finalize_data(  # type:ignore[union-attr]
                dataset_model, dataset, is_full_model=is_full_model, as_global=True
            )
