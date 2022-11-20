"""This module contains the dataset group."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import Literal

import xarray as xr
from attrs import define
from attrs import field

from glotaran.model.dataset_model import DatasetModel
from glotaran.model.dataset_model import get_dataset_model_model_dimension
from glotaran.model.dataset_model import has_dataset_model_global_model
from glotaran.model.item import ModelItem
from glotaran.model.item import fill_item
from glotaran.model.item import item

if TYPE_CHECKING:
    from glotaran.model.model import Model
    from glotaran.parameter import Parameters


@item
class DatasetGroupModel(ModelItem):
    """A group of datasets which will evaluated independently."""

    residual_function: Literal[
        "variable_projection", "non_negative_least_squares"
    ] = "variable_projection"
    """The residual function to use."""

    link_clp: bool | None = None
    """Whether to link the clp parameter."""


@define
class DatasetGroup:
    """A dataset group for optimization."""

    residual_function: Literal["variable_projection", "non_negative_least_squares"]
    """The residual function to use."""

    link_clp: bool | None
    """Whether to link the clp parameter."""

    model: Model
    parameters: Parameters | None = None

    dataset_models: dict[str, DatasetModel] = field(factory=dict)

    def set_parameters(self, parameters: Parameters):
        """Set the group parameters.

        Parameters
        ----------
        parameters : Parameters
            The parameters.
        """
        self.parameters = parameters
        for label in self.dataset_models:
            self.dataset_models[label] = fill_item(
                self.model.dataset[label], self.model, parameters
            )

    def is_linkable(self, parameters: Parameters, data: Mapping[str, xr.Dataset]) -> bool:
        """Check if the group is linkable.

        Parameters
        ----------
        parameters : Parameters
            A parameter set parameters.
        data : Mapping[str, xr.Dataset]
            A the data to link.

        Returns
        -------
        bool
        """
        if any(has_dataset_model_global_model(d) for d in self.dataset_models.values()):
            return False
        dataset_models = [
            fill_item(self.model.dataset[label], self.model, parameters)
            for label in self.dataset_models
        ]
        model_dimensions = {get_dataset_model_model_dimension(d) for d in dataset_models}
        if len(model_dimensions) != 1:
            return False
        global_dimensions = set()
        for dataset in data.values():
            global_dimensions |= {
                dim for dim in dataset.data.coords if dim not in model_dimensions
            }
        return len(global_dimensions) == 1
