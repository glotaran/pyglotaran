from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Literal

import xarray as xr

from glotaran.model.dataset_model import DatasetModel
from glotaran.model_new.item import item

if TYPE_CHECKING:
    from glotaran.model.model import Model
    from glotaran.parameter import ParameterGroup


@item
class DatasetGroupModel:
    """A group of datasets which will evaluated independently."""

    residual_function: Literal[
        "variable_projection", "non_negative_least_squares"
    ] = "variable_projection"
    """The residual function to use."""

    link_clp: bool | None = None
    """Whether to link the clp parameter."""


@dataclass
class DatasetGroup:
    """A dataset group for optimization."""

    residual_function: Literal["variable_projection", "non_negative_least_squares"]
    """The residual function to use."""

    link_clp: bool
    """Whether to link the clp parameter."""

    model: Model
    parameters: ParameterGroup | None = None

    dataset_models: dict[str, DatasetModel] = field(default_factory=dict)

    def set_parameters(self, parameters: ParameterGroup):
        self.parameters = parameters
        for label in self.dataset_models:
            self.dataset_models[label] = self.model.dataset[label].fill(self.model, parameters)

    def is_linkable(self, parameters: ParameterGroup, data: dict[str, xr.Dataset]) -> bool:
        if any(d.has_global_model() for d in self.dataset_models.values()):
            return False
        dataset_models = [
            self.model.dataset[label].fill(self.model, parameters) for label in self.dataset_models
        ]
        model_dimensions = {d.get_model_dimension() for d in dataset_models}
        if len(model_dimensions) != 1:
            return False
        global_dimensions = set()
        for dataset in data.values():
            global_dimensions |= {
                dim for dim in dataset.data.coords if dim not in model_dimensions
            }
        return len(global_dimensions) == 1
