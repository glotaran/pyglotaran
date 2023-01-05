"""This module contains the megacomplex."""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING
from typing import ClassVar

import numpy as np
import xarray as xr

from glotaran.model.item_new import LibraryItemTyped
from glotaran.plugin_system.megacomplex_registration import register_megacomplex

if TYPE_CHECKING:

    from glotaran.model.data_model import DataModel


class Megacomplex(LibraryItemTyped, abc.ABC):  # type:ignore[misc]
    """A base class for megacomplex models.

    Subclasses must overwrite :method:`glotaran.model.Megacomplex.calculate_matrix`.
    """

    data_model_type: ClassVar[type | None] = None
    is_exclusive: ClassVar[bool] = False
    is_unique: ClassVar[bool] = False
    register_as: ClassVar[str | None] = None

    dimension: str | None = None

    def __init_subclass__(cls):
        """Register the megacomplex if necessary."""
        super().__init_subclass__()
        if cls.register_as is not None:
            register_megacomplex(cls.register_as, cls)

    @abc.abstractmethod
    def calculate_matrix(
        self,
        model: DataModel,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
        **kwargs,
    ) -> tuple[list[str], np.typing.ArrayLike]:
        """Calculate the megacomplex matrix.

        Parameters
        ----------
        data_model: DataModel
            The data model.
        global_axis: np.typing.ArrayLike
            The global axis.
        model_axis: np.typing.ArrayLike,
            The model axis.
        **kwargs
            Additional arguments.

        Returns
        -------
        tuple[list[str], np.typing.ArrayLike]:
            The clp labels and the matrix.

        .. # noqa: DAR202
        """
        pass

    def finalize_data(
        self,
        model: DataModel,
        data: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        """Finalize the result data.

        Parameters
        ----------
        data_model: DataModel
            The data model.
        data: xr.Dataset
            The data.
        is_full_model: bool
            Whether the model is a full model.
        as_global: bool
            Whether megacomplex is calculated as global megacomplex.
        """
        pass