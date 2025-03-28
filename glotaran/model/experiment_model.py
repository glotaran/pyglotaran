"""This module contains the dataset group."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SerializationInfo
from pydantic import field_serializer

from glotaran.model.clp_penalties import EqualAreaPenalty  # noqa: TC001
from glotaran.model.clp_relation import ClpRelation  # noqa: TC001
from glotaran.model.data_model import DataModel
from glotaran.model.data_model import resolve_data_model
from glotaran.model.item import ParameterType
from glotaran.model.item import get_item_issues
from glotaran.model.item import resolve_item_parameters
from glotaran.model.item import resolve_parameter
from glotaran.parameter import Parameters
from glotaran.utils.io import serialization_info_to_kwargs

if TYPE_CHECKING:
    from glotaran.model.errors import ItemIssue
    from glotaran.project.library import ModelLibrary


class ExperimentModel(BaseModel):
    """A dataset group for optimization."""

    model_config = ConfigDict(extra="forbid")

    clp_link_tolerance: float = 0.0
    clp_link_method: Literal["nearest", "backward", "forward"] = "nearest"
    clp_penalties: list[EqualAreaPenalty] = Field(default_factory=list)
    clp_relations: list[ClpRelation] = Field(default_factory=list)
    datasets: dict[str, DataModel]
    residual_function: Literal["variable_projection", "non_negative_least_squares"] = Field(
        default="variable_projection", description="The residual function to use."
    )
    scale: dict[str, ParameterType] = Field(
        default_factory=dict,
        description="The scales of of the datasets in the experiment.",
    )

    @field_serializer("datasets")
    def serialize_datasets(
        self, datasets: dict[str, DataModel], info: SerializationInfo
    ) -> dict[str, Any]:
        return {
            name: data_model.model_dump(**serialization_info_to_kwargs(info))
            for name, data_model in datasets.items()
        }

    @property
    def dataset_paths(self) -> dict[str, str]:
        """Paths to all the datasets."""
        return {
            name: data_model.data_path
            for name, data_model in self.datasets.items()
            if data_model.data_path is not None
        }

    @classmethod
    def from_dict(cls, library: ModelLibrary, model_dict: dict[str, Any]) -> ExperimentModel:
        initialized_datasets = {
            label: DataModel.from_dict(library, dataset)
            for label, dataset in model_dict.get("datasets", {}).items()
        }
        return cls.model_validate(model_dict | {"datasets": initialized_datasets})

    def resolve(
        self,
        library: ModelLibrary,
        parameters: Parameters,
        initial: Parameters | None = None,
    ) -> ExperimentModel:
        result = self.model_copy()
        result.datasets = {
            label: resolve_data_model(dataset, library, parameters, initial)
            for label, dataset in self.datasets.items()
        }
        result.clp_penalties = [
            resolve_item_parameters(i, parameters, initial) for i in self.clp_penalties
        ]
        result.clp_relations = [
            resolve_item_parameters(i, parameters, initial) for i in self.clp_relations
        ]
        assert isinstance(initial, Parameters)
        result.scale = {
            label: resolve_parameter(parameter, parameters, initial)
            for label, parameter in self.scale.items()
        }
        return result

    def get_issues(self, parameters: Parameters) -> list[ItemIssue]:
        return [
            issue
            for dataset in self.datasets.values()
            for issue in get_item_issues(dataset, parameters)
        ]


ExperimentModel.model_rebuild()
