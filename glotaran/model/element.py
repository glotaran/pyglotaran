"""This module contains the model."""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal

from glotaran.model.item import TypedItem
from glotaran.plugin_system.element_registration import register_element

if TYPE_CHECKING:
    import xarray as xr

    from glotaran.model.data_model import DataModel
    from glotaran.typing.types import ArrayLike


class Element(TypedItem, abc.ABC):
    """Subclasses must overwrite :method:`glotaran.model.Element.calculate_matrix`."""

    data_model_type: ClassVar[type | None] = None
    is_exclusive: ClassVar[bool] = False
    is_unique: ClassVar[bool] = False
    register_as: ClassVar[str | None] = None

    dimension: str | None = None
    label: str

    def __init_subclass__(cls):
        """Register the model if necessary."""
        super().__init_subclass__()
        if cls.register_as is not None:
            register_element(cls.register_as, cls)

    @abc.abstractmethod
    def calculate_matrix(
        self,
        model: DataModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs,
    ) -> tuple[list[str], ArrayLike]:
        """Calculate the model matrix.

        Parameters
        ----------
        data_model: DataModel
            The data model.
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
        """

    def add_to_result_data(self, model: DataModel, data: xr.Dataset, as_global: bool):
        """

        Parameters
        ----------
        data_model: DataModel
            The data model.
        data: xr.Dataset
            The data.
        is_full_model: bool
            Whether the model is a full model.
        as_global: bool
            Whether model is calculated as global model.
        """


class ExtendableElement(Element):
    extends: list[str] | None = None

    def is_extended(self) -> bool:
        return self.extends is not None

    @abc.abstractmethod
    def extend(self, other: ExtendableElement) -> ExtendableElement:
        pass


class InternalMockElement(Element):
    """An internal model for testing purpose, since at least 2 items
    are needed for pydanticx discriminators.
    """

    type: Literal["internal_mock"]  # type:ignore[assignment]
