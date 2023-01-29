"""This module contains the data model."""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import xarray as xr
from pydantic import Field

from glotaran.model.errors import GlotaranModelError
from glotaran.model.errors import GlotaranUserError
from glotaran.model.errors import ItemIssue
from glotaran.model.item import Attribute
from glotaran.model.item import Item
from glotaran.model.item import LibraryItemType
from glotaran.model.item import ParameterType
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.weight import Weight
from glotaran.parameter import Parameter
from glotaran.parameter import Parameters

if TYPE_CHECKING:
    from glotaran.model.library import Library


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
    value: list[str | Megacomplex] | None, library: Library, is_global: bool
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
        megacomplexes = [library.get_item(Megacomplex, label) for label in labels]
        for megacomplex in megacomplexes:
            megacomplex_type = megacomplex.__class__
            if megacomplex_type.is_exclusive and len(megacomplexes) > 1:
                issues.append(
                    ExclusiveMegacomplexIssue(megacomplex.label, megacomplex.type, is_global)
                )
            if (
                megacomplex_type.is_unique
                and len([m for m in megacomplexes if m.__class__ is megacomplex_type]) > 1
            ):
                issues.append(
                    UniqueMegacomplexIssue(megacomplex.label, megacomplex.type, is_global)
                )
    return issues


def validate_megacomplexes(
    value: list[str | Megacomplex],
    data_model: DataModel,
    library: Library,
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
    return get_megacomplex_issues(value, library, False)


def validate_global_megacomplexes(
    value: list[str | Megacomplex] | None,
    data_model: DataModel,
    library: Library,
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
    return get_megacomplex_issues(value, value, False)


class DataModel(Item):
    """A model for datasets."""

    data: str | xr.Dataset | None = None
    extra_data: str | xr.Dataset | None = None
    megacomplex: list[LibraryItemType[Megacomplex]] = Attribute(
        description="The megacomplexes contributing to this dataset.",
        validator=validate_megacomplexes,  # type:ignore[arg-type]
    )
    megacomplex_scale: list[ParameterType] | None = None
    global_megacomplex: list[LibraryItemType[Megacomplex]] | None = Attribute(
        default=None,
        description="The global megacomplexes contributing to this dataset.",
        validator=validate_global_megacomplexes,  # type:ignore[arg-type]
    )
    global_megacomplex_scale: list[ParameterType] | None = None
    residual_function: Literal["variable_projection", "non_negative_least_squares"] = Attribute(
        default="variable_projection", description="The residual function to use."
    )
    weights: list[Weight] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, library: Library, model_dict: dict[str, Any]) -> DataModel:
        megacomplexes = model_dict.get("megacomplex", None)
        if megacomplexes is None or len(megacomplexes) == 0:
            raise GlotaranModelError("No megcomplex defined for dataset")

        global_megacomplexes = model_dict.get("global_megacomplex", None)
        if global_megacomplexes is not None:
            megacomplexes += global_megacomplexes

        return library.get_data_model_for_megacomplexes(megacomplexes).parse_obj(model_dict)


def is_data_model_global(data_model: DataModel) -> bool:
    """Check if a data model can model the global dimension.

    Parameters
    ----------
    data_model: DataModel
        The data model.

    Returns
    -------
    bool
    """
    return data_model.global_megacomplex is not None and len(data_model.global_megacomplex) != 0


def get_data_model_dimension(data_model: DataModel) -> str:
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
        raise GlotaranModelError(f"No megacomplex set for data model '{data_model.label}'.")
    if any(isinstance(m, str) for m in data_model.megacomplex):
        raise GlotaranUserError(f"Data model '{data_model.label}' was not resolved.")
    model_dimension: str = data_model.megacomplex[
        0
    ].dimension  # type:ignore[union-attr, assignment]
    if any(
        model_dimension != m.dimension  # type:ignore[union-attr]
        for m in data_model.megacomplex
    ):
        raise GlotaranModelError(
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
    is_full_model = is_data_model_global(data_model)
    for megacomplex in data_model.megacomplex:
        megacomplex.finalize_data(  # type:ignore[union-attr]
            data_model, data, is_full_model=is_full_model
        )
    if is_full_model and data_model.global_megacomplex is not None:
        for megacomplex in data_model.global_megacomplex:
            megacomplex.finalize_data(  # type:ignore[union-attr]
                data_model, data, is_full_model=is_full_model, as_global=True
            )
