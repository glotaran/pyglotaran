"""This module contains the data model."""

from __future__ import annotations

from collections.abc import Generator
from collections.abc import Mapping
from typing import Any

import xarray as xr

from glotaran.model.errors import GlotaranDefinitionError
from glotaran.model.errors import GlotaranModelError
from glotaran.model.item_new import Attribute
from glotaran.model.item_new import Item
from glotaran.model.item_new import LibraryItemType
from glotaran.model.item_new import ParameterType
from glotaran.model.megacomplex_new import Megacomplex
from glotaran.model.weight import Weight
from glotaran.parameter import Parameter


class DataModel(Item):
    """A model for datasets."""

    data: str | xr.Dataset | None = None
    extra_data: str | xr.Dataset | None = None
    megacomplex: list[LibraryItemType[Megacomplex]] = Attribute(
        #  validator=validate_megacomplexes  # type:ignore[arg-type]
        description="The megacomplexes contributing to this dataset."
    )
    megacomplex_scale: list[ParameterType] | None = None
    global_megacomplex: list[LibraryItemType[Megacomplex]] | None = Attribute(
        default=None,
        description="The global megacomplexes contributing to this dataset."
        #  validator=validate_global_megacomplexes,  # type:ignore[arg-type]
    )
    global_megacomplex_scale: list[ParameterType] | None = None
    scale: ParameterType | None = None
    weights: list[Weight] | None = None


def get_megacomplex_types_from_data_model_dict(
    data_model_dict: dict[str, Any], megacomplex_registry: Mapping[str, type[Megacomplex]]
) -> set[type[Megacomplex]]:
    megacomplexes = data_model_dict.get("megacomplex", None)
    if megacomplexes is None or len(megacomplexes) == 0:
        raise GlotaranModelError(
            f"No megacomplex defined for datamodel with '{data_model_dict.get('label', None)}'"
        )

    global_megacomplexes = data_model_dict.get("megacomplex", None)
    if global_megacomplexes is not None:
        megacomplexes = megacomplexes + global_megacomplexes

    try:
        return {megacomplex_registry[megacomplex] for megacomplex in megacomplexes}
    except KeyError as e:
        raise GlotaranDefinitionError(f"Unknown megacomplex type '{e}'.") from e


def has_data_model_global_model(data_model: DataModel) -> bool:
    """Check if the data model can model the global dimension.

    Parameters
    ----------
    data_model: DataModel
        The data model.

    Returns
    -------
    bool
    """
    return data_model.global_megacomplex is not None and len(data_model.global_megacomplex) != 0


def get_data_model_model_dimension(data_model: DataModel) -> str:
    """Get the data model's model dimension.

    Parameters
    ----------
    data_model: DataModel
        The data model.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        Raised if the data model does not have megacomplexes or if it is not filled.
    """
    if len(data_model.megacomplex) == 0:
        raise ValueError(f"No megacomplex set for data model '{data_model.label}'.")
    if any(isinstance(m, str) for m in data_model.megacomplex):
        raise ValueError(f"Dataset model '{data_model.label}' was not filled.")
    model_dimension: str = data_model.megacomplex[
        0
    ].dimension  # type:ignore[union-attr, assignment]
    if any(
        model_dimension != m.dimension  # type:ignore[union-attr]
        for m in data_model.megacomplex
    ):
        raise ValueError(
            f"Megacomplex dimensions do not match for data model '{data_model.label}'."
        )
    return model_dimension


def iterate_data_model_megacomplexes(
    data_model: DataModel,
) -> Generator[tuple[Parameter | str | None, Megacomplex | str], None, None]:
    """Iterate the data model's megacomplexes.

    Parameters
    ----------
    data_model: DataModel
        The data model.

    Yields
    ------
    tuple[Parameter | str | None, Megacomplex | str]
        A scale and megacomplex.
    """
    for i, megacomplex in enumerate(data_model.megacomplex):
        scale = (
            data_model.megacomplex_scale[i] if data_model.megacomplex_scale is not None else None
        )
        yield scale, megacomplex


def iterate_data_model_global_megacomplexes(
    data_model: DataModel,
) -> Generator[tuple[Parameter | str | None, Megacomplex | str], None, None]:
    """Iterate the data model's global megacomplexes.

    Parameters
    ----------
    data_model: DataModel
        The data model.

    Yields
    ------
    tuple[Parameter | str | None, Megacomplex | str]
        A scale and megacomplex.
    """
    if data_model.global_megacomplex is None:
        return
    for i, megacomplex in enumerate(data_model.global_megacomplex):
        scale = (
            data_model.global_megacomplex_scale[i]
            if data_model.global_megacomplex_scale is not None
            else None
        )
        yield scale, megacomplex


def finalize_data_model(data_model: DataModel, data: xr.Dataset):
    """Finalize a data by applying all megacomplex finalize methods.

    Parameters
    ----------
    data_model: DataModel
        The data model.
    data: xr.Dataset
        The data.
    """
    is_full_model = has_data_model_global_model(data_model)
    for megacomplex in data_model.megacomplex:
        megacomplex.finalize_data(  # type:ignore[union-attr]
            data_model, data, is_full_model=is_full_model
        )
    if is_full_model and data_model.global_megacomplex is not None:
        for megacomplex in data_model.global_megacomplex:
            megacomplex.finalize_data(  # type:ignore[union-attr]
                data_model, data, is_full_model=is_full_model, as_global=True
            )
