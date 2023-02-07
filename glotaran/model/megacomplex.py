"""This module contains the megacomplex."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import ClassVar

import xarray as xr
from attrs import NOTHING
from attrs import fields

from glotaran.model.item import ModelItemTyped
from glotaran.model.item import item
from glotaran.plugin_system.megacomplex_registration import register_megacomplex

if TYPE_CHECKING:
    from glotaran.model import DatasetModel
    from glotaran.typing.types import ArrayLike


def megacomplex(
    *,
    dataset_model_type: type[DatasetModel] | None = None,
    exclusive: bool = False,
    unique: bool = False,
) -> Callable:
    """Create a megacomplex from a class.

    Parameters
    ----------
    dataset_model_type: type
        The dataset model type.
    exclusive: bool
        Whether the megacomplex is exclusive.
    unique: bool
        Whether the megacomplex is unique.

    Returns
    -------
    Callable
    """

    def decorator(cls):
        megacomplex_type = item(cls)
        megacomplex_type.__dataset_model_type__ = dataset_model_type
        megacomplex_type.__is_exclusive__ = exclusive
        megacomplex_type.__is_unique__ = unique

        megacomplex_type_str = fields(cls).type.default
        if megacomplex_type_str is not NOTHING:
            register_megacomplex(megacomplex_type_str, megacomplex_type)

        return megacomplex_type

    return decorator


@item
class Megacomplex(ModelItemTyped):
    """A base class for megacomplex models.

    Subclasses must overwrite :method:`glotaran.model.Megacomplex.calculate_matrix`
    and :method:`glotaran.model.Megacomplex.index_dependent`.
    """

    dimension: str | None = None

    __dataset_model_type__: ClassVar[type | None] = None
    __is_exclusive__: ClassVar[bool]
    __is_unique__: ClassVar[bool]

    @classmethod
    def get_dataset_model_type(cls) -> type | None:
        """Get the dataset model type.

        Returns
        -------
        type | None
        """
        return cls.__dataset_model_type__

    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs,
    ) -> tuple[list[str], ArrayLike]:
        """Calculate the megacomplex matrix.

        Parameters
        ----------
        dataset_model: DatasetModel
            The dataset model.
        global_axis: ArrayLike
            The global axis.
        model_axis: ArrayLike
            The model axis.
        **kwargs
            Additional arguments.

        Returns
        -------
        tuple[list[str], ArrayLike]:
            The clp labels and the matrix.

        .. # noqa: DAR202
        .. # noqa: DAR401
        """
        raise NotImplementedError

    def finalize_data(
        self,
        dataset_model: DatasetModel,
        dataset: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        """Finalize a dataset.

        Parameters
        ----------
        dataset_model: DatasetModel
            The dataset model.
        dataset: xr.Dataset
            The dataset.
        is_full_model: bool
            Whether the model is a full model.
        as_global: bool
            Whether megacomplex is calculated as global megacomplex.


        .. # noqa: DAR101
        .. # noqa: DAR401
        """
        raise NotImplementedError


def is_exclusive(cls: type[Megacomplex]) -> bool:
    """Check if the megacomplex is exclusive.

    Parameters
    ----------
    cls: type[Megacomplex]
        The megacomplex type.

    Returns
    -------
    bool
    """
    return cls.__is_exclusive__


def is_unique(cls: type[Megacomplex]) -> bool:
    """Check if the megacomplex is unique.

    Parameters
    ----------
    cls: type[Megacomplex]
        The megacomplex type.

    Returns
    -------
    bool
    """
    return cls.__is_unique__
