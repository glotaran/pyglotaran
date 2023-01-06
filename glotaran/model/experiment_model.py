"""This module contains the dataset group."""
from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Extra
from pydantic import Field

from glotaran.model.clp_constraint import ClpConstraint
from glotaran.model.clp_penalties import EqualAreaPenalty
from glotaran.model.clp_relation import ClpRelation
from glotaran.model.data_model import DataModel
from glotaran.model.library import Library
from glotaran.model.weight import Weight
from glotaran.parameter import Parameter


class ExperimentModel(BaseModel):
    """A dataset group for optimization."""

    class Config:
        """Config for pydantic.BaseModel."""

        arbitrary_types_allowed = True
        extra = Extra.forbid

    clp_constraints: list[ClpConstraint.get_annotated_type()] = Field(default_factory=list)
    clp_penalties: list[EqualAreaPenalty] = Field(default_factory=list)
    clp_relations: list[ClpRelation] = Field(default_factory=list)
    datasets: dict[str, DataModel]
    link_clp: bool | None = Field(None, description="Whether to link the clp.")
    residual_function: Literal["variable_projection", "non_negative_least_squares"] = Field(
        "variable_projection", description="The residual function to use."
    )
    weights: list[Weight] = Field(default_factory=list)
    scale: dict[str, Parameter] = Field(
        default_factory=dict, description="The scales of of the datasets in the experiment."
    )

    @classmethod
    def from_dict(cls, library: Library, model_dict: dict[str, Any]) -> ExperimentModel:
        model_dict["datasets"] = {
            label: DataModel.from_dict(library, dataset)
            for label, dataset in model_dict.get("datasets", {}).items()
        }
        return cls.parse_obj(model_dict)
