"""This module contains the model."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from pydantic import ConfigDict
from pydantic import Field

from glotaran.model.clp_constraint import ClpConstraint  # noqa: TCH001
from glotaran.model.item import TypedItem
from glotaran.plugin_system.element_registration import register_element

if TYPE_CHECKING:
    import xarray as xr

    from glotaran.model.data_model import DataModel
    from glotaran.typing.types import ArrayLike


def _sanitize_json_schema(json_schema: dict[str, Any]) -> None:
    """Remove internal attribute ``label`` from schema dict.

    Parameters
    ----------
    json_schema : dict[str, Any]
        Json Schema generated by pydantic.
    """
    json_schema["properties"].pop("label")
    json_schema["required"].remove("label")


@dataclass
class ElementResult:
    amplitudes: dict[str, xr.DataArray]
    concentrations: dict[str, xr.DataArray]
    extra: dict[str, xr.DataArray] = field(default_factory=dict)


class Element(TypedItem, abc.ABC):
    """Subclasses must overwrite :method:`glotaran.model.Element.calculate_matrix`."""

    data_model_type: ClassVar[type | None] = None
    is_exclusive: ClassVar[bool] = False
    is_unique: ClassVar[bool] = False
    register_as: ClassVar[str | None] = None

    dimension: str | None = None
    label: str
    clp_constraints: list[ClpConstraint.get_annotated_type()] = (  # type:ignore[valid-type]
        Field(default_factory=list)
    )

    model_config = ConfigDict(json_schema_extra=_sanitize_json_schema)

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

    def create_result(
        self,
        model: DataModel,
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.Dataset,
        concentrations: xr.Dataset,
    ) -> ElementResult:
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
        return ElementResult({}, {})


class ExtendableElement(Element):
    extends: list[str] | None = None

    def is_extended(self) -> bool:
        return self.extends is not None

    @abc.abstractmethod
    def extend(self, other: ExtendableElement) -> ExtendableElement:
        pass
