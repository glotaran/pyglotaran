"""This module contains the dataset group."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import Field

from glotaran.model.clp_constraint import ClpConstraint
from glotaran.model.clp_penalties import ClpPenalty
from glotaran.model.clp_relation import ClpRelation
from glotaran.model.dataset_model import DatasetModel


class ExperimentModel(BaseModel):
    """A dataset group for optimization."""

    clp_constraints: list[ClpConstraint] = Field(..., default_factory=list)
    clp_penalties: list[ClpPenalty] = Field(..., default_factory=list)
    clp_relations: list[ClpRelation] = Field(..., default_factory=list)
    datasets: dict[str, DatasetModel | str]
    link_clp: bool | None = Field(None, description="Whether to link the clp.")
    residual_function: Literal["variable_projection", "non_negative_least_squares"] = Field(
        "variable_projection", description="The residual function to use."
    )
