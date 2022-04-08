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
    model: DatasetGroupModel
    dataset_models: dict[str, DatasetModel] = field(default_factory=dict)

    def fill(self, model: Model, parameters: ParameterGroup):
        for label, dataset_model in self.dataset_models.items():
            self.dataset_models[label] = dataset_model.fill(model, parameters)
