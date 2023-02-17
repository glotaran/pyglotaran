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
from glotaran.model.data_model import resolve_data_model
from glotaran.model.element import Element
from glotaran.model.errors import ItemIssue
from glotaran.model.item import get_item_issues
from glotaran.parameter import Parameter
from glotaran.parameter import Parameters


class ExperimentModel(BaseModel):
    """A dataset group for optimization."""

    class Config:
        """Config for pydantic.BaseModel."""

        arbitrary_types_allowed = True
        extra = Extra.forbid

    clp_link_tolerance: float = 0.0
    clp_link_method: Literal["nearest", "backward", "forward"] = "nearest"
    clp_constraints: list[ClpConstraint.get_annotated_type()] = Field(default_factory=list)
    clp_penalties: list[EqualAreaPenalty] = Field(default_factory=list)
    clp_relations: list[ClpRelation] = Field(default_factory=list)
    datasets: dict[str, DataModel]
    residual_function: Literal["variable_projection", "non_negative_least_squares"] = Field(
        "variable_projection", description="The residual function to use."
    )
    scale: dict[str, Parameter] = Field(
        default_factory=dict, description="The scales of of the datasets in the experiment."
    )

    @classmethod
    def from_dict(cls, library: dict[str, Element], model_dict: dict[str, Any]) -> ExperimentModel:
        model_dict["datasets"] = {
            label: DataModel.from_dict(library, dataset)
            for label, dataset in model_dict.get("datasets", {}).items()
        }
        return cls.parse_obj(model_dict)

    def resolve(
        self,
        library: dict[str, Element],
        parameters: Parameters,
        initial: Parameters | None = None,
    ) -> ExperimentModel:
        result = self.copy()
        result.datasets = {
            label: resolve_data_model(dataset, library, parameters, initial)
            for label, dataset in self.datasets.items()
        }
        return result

    def get_issues(self, parameters: Parameters) -> list[ItemIssue]:
        return [
            issue
            for dataset in self.datasets.values()
            for issue in get_item_issues(dataset, parameters)
        ]
