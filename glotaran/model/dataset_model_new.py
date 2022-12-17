"""This module contains the dataset model."""

from __future__ import annotations

from collections.abc import Generator

import xarray as xr

from glotaran.model.item_new import Attribute
from glotaran.model.item_new import Item
from glotaran.model.item_new import LibraryItemType
from glotaran.model.item_new import ParameterType
from glotaran.model.megacomplex_new import Megacomplex
from glotaran.model.weight import Weight
from glotaran.parameter import Parameter


class DatasetModel(Item):
    """A model for datasets."""

    data: str | xr.Dataset | None = None
    extra_data: str | xr.Dataset | None = None
    megacomplex: list[LibraryItemType[Megacomplex]] = Attribute(
        #  validator=validate_megacomplexes  # type:ignore[arg-type]
        description="The megacomplexes contributing to this dataset"
    )
    megacomplex_scale: list[ParameterType] | None = None
    global_megacomplex: list[LibraryItemType[Megacomplex]] | None = Attribute(
        default=None,
        description="The global megacomplexes contributing to this dataset"
        #  validator=validate_global_megacomplexes,  # type:ignore[arg-type]
    )
    global_megacomplex_scale: list[ParameterType] | None = None
    scale: ParameterType | None = None
    weights: list[Weight] | None = None


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
