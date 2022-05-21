from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Literal

from glotaran.model.dataset_model import DatasetModel

if TYPE_CHECKING:
    from glotaran.model.model import Model
    from glotaran.parameter import ParameterGroup


@dataclass
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
