from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

import numpy as np
import xarray as xr
from attrs import NOTHING
from attrs import fields

from glotaran.model.item import ModelItemTyped
from glotaran.model.item import item
from glotaran.plugin_system.megacomplex_registration import register_megacomplex

if TYPE_CHECKING:

    from glotaran.model import DatasetModel


def megacomplex(
    *,
    dataset_model_type: type | None = None,
    exclusive: bool = False,
    unique: bool = False,
):
    """The `@megacomplex` decorator is intended to be used on subclasses of
    :class:`glotaran.model.Megacomplex`. It registers the megacomplex model
    and makes it available in analysis models.
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
        return cls.__dataset_model_type__

    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        global_index: int | None,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
        **kwargs,
    ) -> xr.DataArray:
        raise NotImplementedError

    def index_dependent(self, dataset_model: DatasetModel) -> bool:
        raise NotImplementedError

    def finalize_data(
        self,
        dataset_model: DatasetModel,
        dataset: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        raise NotImplementedError


def is_exclusive(cls: type[Megacomplex]) -> bool:
    return cls.__is_exclusive__


def is_unique(cls: type[Megacomplex]) -> bool:
    return cls.__is_unique__
