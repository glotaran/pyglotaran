from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

import numpy as np
import xarray as xr

from glotaran.model_new.item import ModelItemTyped
from glotaran.model_new.item import item
from glotaran.plugin_system.megacomplex_registration import register_megacomplex

if TYPE_CHECKING:

    from glotaran.model import DatasetModel


def megacomplex(
    *,
    dataset_model_type: type | None = None,
    dimension: str | None = None,
    unique: bool = False,
    exclusive: bool = False,
    register_as: str | None = None,
):
    """The `@megacomplex` decorator is intended to be used on subclasses of
    :class:`glotaran.model.Megacomplex`. It registers the megacomplex model
    and makes it available in analysis models.
    """

    def decorator(cls):
        setattr(cls, "_glotaran_megacomplex_unique", unique)
        setattr(cls, "_glotaran_megacomplex_exclusive", exclusive)

        megacomplex_type = item(cls)
        if dataset_model_type is not None:
            megacomplex_type.__dataset_model_type__ = dataset_model_type

        if register_as is not None:
            megacomplex_type.name = register_as
            register_megacomplex(register_as, megacomplex_type)

        return megacomplex_type

    return decorator


@item
class Megacomplex(ModelItemTyped):
    """A base class for megacomplex models.

    Subclasses must overwrite :method:`glotaran.model.Megacomplex.calculate_matrix`
    and :method:`glotaran.model.Megacomplex.index_dependent`.
    """

    __dataset_model_type__: ClassVar[type | None] = None

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

    @classmethod
    def glotaran_unique(cls) -> bool:
        return cls._glotaran_megacomplex_unique

    @classmethod
    def glotaran_exclusive(cls) -> bool:
        return cls._glotaran_megacomplex_exclusive
