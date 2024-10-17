"""This module contains the data model."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from uuid import uuid4

import xarray as xr  # noqa: TCH002
from pydantic import Field
from pydantic import create_model

from glotaran.model.element import Element
from glotaran.model.errors import GlotaranModelError
from glotaran.model.errors import GlotaranUserError
from glotaran.model.errors import ItemIssue
from glotaran.model.item import Attribute
from glotaran.model.item import Item
from glotaran.model.item import ParameterType
from glotaran.model.item import resolve_item_parameters
from glotaran.model.weight import Weight  # noqa: TCH001

if TYPE_CHECKING:
    from collections.abc import Generator

    from glotaran.parameter import Parameters
    from glotaran.project.library import ModelLibrary


class ExclusiveModelIssue(ItemIssue):
    """Issue for exclusive elements."""

    def __init__(self, label: str, element_type: str, is_global: bool):
        """Create an ExclusiveModelIssue.

        Parameters
        ----------
        label : str
            The element label.
        element_type : str
            The element type.
        is_global : bool
            Whether the element is global.
        """
        self._label = label
        self._type = element_type
        self._is_global = is_global

    def to_string(self) -> str:
        """Get the issue as string.

        Returns
        -------
        str
        """
        return (
            f"Exclusive {'global ' if self._is_global else ''}element '{self._label}' of "
            f"type '{self._type}' cannot be combined with other elements."
        )


class UniqueModelIssue(ItemIssue):
    """Issue for unique elements."""

    def __init__(self, label: str, element_type: str, is_global: bool):
        """Create a UniqueModelIssue.

        Parameters
        ----------
        label : str
            The element label.
        element_type : str
            The element type.
        is_global : bool
            Whether the element is global.
        """
        self._label = label
        self._type = element_type
        self._is_global = is_global

    def to_string(self):
        """Get the issue as string.

        Returns
        -------
        str
        """
        return (
            f"Unique {'global ' if self._is_global else ''}element '{self._label}' of "
            f"type '{self._type}' can only be used once per dataset."
        )


def get_element_issues(value: list[str | Element] | None, is_global: bool) -> list[ItemIssue]:
    """Get issues for elements.

    Parameters
    ----------
    value: list[str | Element] | None
        A list of elements.
    element: Element
        The element.
    is_global: bool
        Whether the elements are global.

    Returns
    -------
    list[ItemIssue]
    """
    issues: list[ItemIssue] = []

    if value is not None:
        elements = [v for v in value if isinstance(v, Element)]
        for element in elements:
            element_type = element.__class__
            if element_type.is_exclusive and len(elements) > 1:
                issues.append(
                    ExclusiveModelIssue(element.label, element.type, is_global)  # type:ignore[arg-type]
                )
            if (
                element_type.is_unique
                and len([m for m in elements if m.__class__ is element_type]) > 1
            ):
                issues.append(UniqueModelIssue(element.label, element.type, is_global))  # type:ignore[arg-type]
    return issues


def validate_elements(
    value: list[str | Element],
    data_model: DataModel,
    parameters: Parameters | None,
) -> list[ItemIssue]:
    """Get issues for dataset model elements.

    Parameters
    ----------
    value: list[str | Element]
        A list of elements.
    dataset_model: DatasetModel
        The dataset model.
    element: Element
        The element.
    parameters: Parameters | None,
        The parameters.

    Returns
    -------
    list[ItemIssue]
    """
    return get_element_issues(value, False)


def validate_global_elements(
    value: list[str | Element] | None,
    data_model: DataModel,
    parameters: Parameters | None,
) -> list[ItemIssue]:
    """Get issues for dataset model global elements.

    Parameters
    ----------
    value: list[str | Element] | None
        A list of elements.
    dataset_model: DatasetModel
        The dataset model.
    element: Element
        The element.
    parameters: Parameters | None,
        The parameters.

    Returns
    -------
    list[ItemIssue]
    """
    return get_element_issues(value, True)


class DataModel(Item):
    """A model for datasets."""

    data: str | xr.Dataset | None = None
    # Seems unused:
    # extra_data: str | xr.Dataset | None = None
    elements: list[Element | str] = Attribute(
        description="The elements contributing to this dataset.",
        validator=validate_elements,
    )
    element_scale: dict[str, ParameterType] | None = None
    global_elements: list[Element | str] | None = Attribute(
        default=None,
        description="The global elements contributing to this dataset.",
        validator=validate_global_elements,
    )
    global_element_scale: dict[str, ParameterType] | None = None
    residual_function: Literal["variable_projection", "non_negative_least_squares"] = Attribute(
        default="variable_projection", description="The residual function to use."
    )
    weights: list[Weight] = Field(default_factory=list)

    @staticmethod
    def create_class_for_elements(elements: set[type[Element]]) -> type[DataModel]:
        data_model_cls_name = f"GlotaranDataModel_{str(uuid4()).replace('-','_')}"
        data_models = (
            *tuple({e.data_model_type for e in elements if e.data_model_type is not None}),
            DataModel,
        )
        return create_model(data_model_cls_name, __base__=data_models)

    @classmethod
    def from_dict(cls, library: ModelLibrary, model_dict: dict[str, Any]) -> DataModel:
        element_labels = model_dict.get("elements", []) + model_dict.get("global_elements", [])
        if len(element_labels) == 0:
            raise GlotaranModelError("No element defined for dataset")
        elements = {type(library[label]) for label in element_labels}
        return cls.create_class_for_elements(elements)(**model_dict)

    @staticmethod
    def create_result(
        model: DataModel,
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.DataArray,
        concentrations: xr.DataArray,
    ) -> dict[str, xr.DataArray]:
        return {}


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
    return data_model.global_elements is not None and len(data_model.global_elements) != 0


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
        Raised if the data model does not have elements or if it is not filled.
    """
    if len(data_model.elements) == 0:
        raise GlotaranModelError("No elements set for data model.")
    if any(isinstance(m, str) for m in data_model.elements):
        raise GlotaranUserError("Data model was not resolved.")
    model_dimension: str = data_model.elements[0].dimension  # type:ignore[union-attr, assignment]
    if any(
        model_dimension != m.dimension  # type:ignore[union-attr]
        for m in data_model.elements
    ):
        raise GlotaranModelError("Model dimensions do not match for data model.")
    if model_dimension is None:
        raise GlotaranModelError("No models dimensions defined for data model.")
    return model_dimension


def iterate_data_model_elements(
    data_model: DataModel,
) -> Generator[tuple[ParameterType | None, Element | str], None, None]:
    """Iterate the data model's elements.

    Parameters
    ----------
    data_model: DataModel
        The data model.

    Yields
    ------
    tuple[Parameter | str | None, Element | str]
        A scale and elements.
    """
    scales = data_model.element_scale
    for element in data_model.elements:
        scale = None
        if scales is not None:
            element_label = element if isinstance(element, str) else element.label
            scale = scales.get(element_label, 1)
        yield scale, element


def iterate_data_model_global_elements(
    data_model: DataModel,
) -> Generator[tuple[ParameterType | None, Element | str], None, None]:
    """Iterate the data model's global elements.

    Parameters
    ----------
    data_model: DataModel
        The data model.

    Yields
    ------
    tuple[Parameter | str | None, Element | str]
        A scale and element.
    """
    if data_model.global_elements is None:
        return
    scales = data_model.global_element_scale
    for element in data_model.global_elements:
        scale = None
        if scales is not None:
            element_label = element if isinstance(element, str) else element.label
            scale = scales.get(element_label, 1)
        yield scale, element


def resolve_data_model(
    model: DataModel,
    library: ModelLibrary,
    parameters: Parameters,
    initial: Parameters | None = None,
) -> DataModel:
    model = model.model_copy()
    model.elements = [library[m] if isinstance(m, str) else m for m in model.elements]
    if model.global_elements is not None:
        model.global_elements = [
            library[m] if isinstance(m, str) else m for m in model.global_elements
        ]
    return resolve_item_parameters(model, parameters, initial)
