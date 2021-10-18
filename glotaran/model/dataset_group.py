from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Literal

from glotaran.model.dataset_model import DatasetModel


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
