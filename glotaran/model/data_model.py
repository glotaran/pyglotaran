"""This module contains the data model."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from typing import Literal
from uuid import uuid4

import xarray as xr
from pydantic import Field
from pydantic import create_model

from glotaran.model.errors import GlotaranModelError
from glotaran.model.errors import GlotaranUserError
from glotaran.model.errors import ItemIssue
from glotaran.model.item import Attribute
from glotaran.model.item import Item
from glotaran.model.item import ParameterType
from glotaran.model.item import resolve_item_parameters
from glotaran.model.model import Model
from glotaran.model.weight import Weight
from glotaran.parameter import Parameter
from glotaran.parameter import Parameters


class ExclusiveModelIssue(ItemIssue):
    """Issue for exclusive models."""

    def __init__(self, label: str, model_type: str, is_global: bool):
        """Create an ExclusiveModelIssue.

        Parameters
        ----------
        label : str
            The model label.
        model_type : str
            The model type.
        is_global : bool
            Whether the model is global.
        """
        self._label = label
        self._type = model_type
        self._is_global = is_global

    def to_string(self) -> str:
        """Get the issue as string.

        Returns
        -------
        str
        """
        return (
            f"Exclusive {'global ' if self._is_global else ''}model '{self._label}' of "
            f"type '{self._type}' cannot be combined with other models."
        )


class UniqueModelIssue(ItemIssue):
    """Issue for unique models."""

    def __init__(self, label: str, model_type: str, is_global: bool):
        """Create a UniqueModelIssue.

        Parameters
        ----------
        label : str
            The model label.
        model_type : str
            The model type.
        is_global : bool
            Whether the model is global.
        """
        self._label = label
        self._type = model_type
        self._is_global = is_global

    def to_string(self):
        """Get the issue as string.

        Returns
        -------
        str
        """
        return (
            f"Unique {'global ' if self._is_global else ''}model '{self._label}' of "
            f"type '{self._type}' can only be used once per dataset."
        )


def get_model_issues(value: list[str | Model] | None, is_global: bool) -> list[ItemIssue]:
    """Get issues for models.

    Parameters
    ----------
    value: list[str | Model] | None
        A list of models.
    model: Model
        The model.
    is_global: bool
        Whether the models are global.

    Returns
    -------
    list[ItemIssue]
    """
    issues: list[ItemIssue] = []

    if value is not None:
        models = [v for v in value if isinstance(v, Model)]
        for model in models:
            model_type = model.__class__
            if model_type.is_exclusive and len(models) > 1:
                issues.append(ExclusiveModelIssue(model.label, model.type, is_global))
            if model_type.is_unique and len([m for m in models if m.__class__ is model_type]) > 1:
                issues.append(UniqueModelIssue(model.label, model.type, is_global))
    return issues


def validate_models(
    value: list[str | Model],
    data_model: DataModel,
    parameters: Parameters | None,
) -> list[ItemIssue]:
    """Get issues for dataset model models.

    Parameters
    ----------
    value: list[str | Model]
        A list of models.
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
    return get_model_issues(value, False)


def validate_global_models(
    value: list[str | Model] | None,
    data_model: DataModel,
    parameters: Parameters | None,
) -> list[ItemIssue]:
    """Get issues for dataset model global models.

    Parameters
    ----------
    value: list[str | Model] | None
        A list of models.
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
    return get_model_issues(value, True)


class DataModel(Item):
    """A model for datasets."""

    data: str | xr.Dataset | None = None
    extra_data: str | xr.Dataset | None = None
    models: list[Model | str] = Attribute(
        description="The models contributing to this dataset.",
        validator=validate_models,  # type:ignore[arg-type]
    )
    model_scale: list[ParameterType] | None = None
    global_models: list[Model | str] | None = Attribute(
        default=None,
        description="The global models contributing to this dataset.",
        validator=validate_global_models,  # type:ignore[arg-type]
    )
    global_model_scale: list[ParameterType] | None = None
    residual_function: Literal["variable_projection", "non_negative_least_squares"] = Attribute(
        default="variable_projection", description="The residual function to use."
    )
    weights: list[Weight] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, library: dict[str, Model], model_dict: dict[str, Any]) -> DataModel:
        data_model_cls_name = f"GlotaranDataModel_{str(uuid4()).replace('-','_')}"
        model_labels = model_dict.get("models", []) + model_dict.get("global_models", [])
        if len(model_labels) == 0:
            raise GlotaranModelError("No model defined for dataset")
        models = {type(library[label]) for label in model_labels}
        data_models = [
            m.data_model_type for m in filter(lambda m: m.data_model_type is not None, models)
        ] + [DataModel]
        return create_model(data_model_cls_name, __base__=tuple(data_models))(**model_dict)


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
    return data_model.global_models is not None and len(data_model.global_models) != 0


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
        Raised if the data model does not have models or if it is not filled.
    """
    if len(data_model.models) == 0:
        raise GlotaranModelError(f"No models set for data model '{data_model.label}'.")
    if any(isinstance(m, str) for m in data_model.models):
        raise GlotaranUserError(f"Data model '{data_model.label}' was not resolved.")
    model_dimension: str = data_model.models[0].dimension  # type:ignore[union-attr, assignment]
    if any(
        model_dimension != m.dimension  # type:ignore[union-attr]
        for m in data_model.models
    ):
        raise GlotaranModelError("Model dimensions do not match for data model.")
    if model_dimension is None:
        raise GlotaranModelError("No models dimensions defined for data model.")
    return model_dimension


def iterate_data_model_models(
    data_model: DataModel,
) -> Generator[tuple[Parameter | str | None, Model | str], None, None]:
    """Iterate the data model's models.

    Parameters
    ----------
    data_model: DataModel
        The data model.

    Yields
    ------
    tuple[Parameter | str | None, Model | str]
        A scale and models.
    """
    for i, model in enumerate(data_model.models):
        scale = data_model.models_scale[i] if data_model.model_scale is not None else None
        yield scale, model


def iterate_data_model_global_models(
    data_model: DataModel,
) -> Generator[tuple[Parameter | str | None, Model | str], None, None]:
    """Iterate the data model's global models.

    Parameters
    ----------
    data_model: DataModel
        The data model.

    Yields
    ------
    tuple[Parameter | str | None, Model | str]
        A scale and model.
    """
    if data_model.global_models is None:
        return
    for i, model in enumerate(data_model.global_models):
        scale = (
            data_model.global_model_scale[i] if data_model.global_model_scale is not None else None
        )
        yield scale, model


def resolve_data_model(
    model: DataModel,
    library: dict[str, Model],
    parameters: Parameters,
    initial: Parameters | None = None,
) -> DataModel:
    model = model.copy()
    model.models = [library[m] if isinstance(m, str) else m for m in model.models]
    if model.global_models is not None:
        model.global_models = [
            library[m] if isinstance(m, str) else m for m in model.global_models
        ]
    return resolve_item_parameters(model, parameters, initial)


def finalize_data_model(data_model: DataModel, data: xr.Dataset):
    """Finalize a data by applying all model finalize methods.

    Parameters
    ----------
    data_model: DataModel
        The data model.
    data: xr.Dataset
        The data.
    """
    is_full_model = is_data_model_global(data_model)
    for model in data_model.models:
        model.finalize_data(  # type:ignore[union-attr]
            data_model, data, is_full_model=is_full_model
        )
    if is_full_model and data_model.global_models is not None:
        for model in data_model.global_models:
            model.finalize_data(  # type:ignore[union-attr]
                data_model, data, is_full_model=is_full_model, as_global=True
            )
