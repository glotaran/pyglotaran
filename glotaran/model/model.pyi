from typing import Any

from _typeshed import Incomplete

from glotaran.deprecation import raise_deprecation_error
from glotaran.io import load_model
from glotaran.model.clp_penalties import EqualAreaPenalty
from glotaran.model.constraint import Constraint
from glotaran.model.dataset_group import DatasetGroup
from glotaran.model.dataset_group import DatasetGroupModel
from glotaran.model.dataset_model import DatasetModel
from glotaran.model.dataset_model import create_dataset_model_type
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.megacomplex import create_model_megacomplex_type
from glotaran.model.relation import Relation
from glotaran.model.util import ModelError
from glotaran.model.weight import Weight
from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup
from glotaran.plugin_system.megacomplex_registration import get_megacomplex
from glotaran.utils.ipython import MarkdownStr

default_model_items: Incomplete
default_dataset_properties: Incomplete
root_parameter_error: Incomplete

class Model:
    loader: Incomplete
    source_path: str
    def __init__(
        self,
        *,
        megacomplex_types: dict[str, type[Megacomplex]],
        default_megacomplex_type: str | None = ...,
        dataset_group_models: dict[str, DatasetGroupModel] = ...,
    ) -> None: ...
    @classmethod
    def from_dict(
        cls,
        model_dict: dict[str, Any],
        *,
        megacomplex_types: dict[str, type[Megacomplex]] | None = ...,
        default_megacomplex_type: str | None = ...,
    ) -> Model: ...
    @property
    def model_dimension(self) -> None: ...
    @property
    def global_dimension(self) -> None: ...
    @property
    def default_megacomplex(self) -> str: ...
    @property
    def megacomplex_types(self) -> dict[str, type[Megacomplex]]: ...
    @property
    def dataset_group_models(self) -> dict[str, DatasetGroupModel]: ...
    @property
    def model_items(self) -> dict[str, type[object]]: ...
    @property
    def global_megacomplex(self) -> dict[str, Megacomplex]: ...
    def get_dataset_groups(self) -> dict[str, DatasetGroup]: ...
    def as_dict(self) -> dict: ...
    def get_parameter_labels(self) -> list[str]: ...
    def generate_parameters(self) -> dict | list: ...
    def need_index_dependent(self) -> bool: ...
    def problem_list(self, parameters: ParameterGroup | None = ...) -> list[str]: ...
    def validate(
        self, parameters: ParameterGroup = ..., raise_exception: bool = ...
    ) -> MarkdownStr: ...
    def valid(self, parameters: ParameterGroup = ...) -> bool: ...
    def markdown(
        self,
        parameters: ParameterGroup = ...,
        initial_parameters: ParameterGroup = ...,
        base_heading_level: int = ...,
    ) -> MarkdownStr: ...
    @property
    def clp_area_penalties(self) -> dict[str, EqualAreaPenalty]: ...
    @property
    def clp_constraints(self) -> dict[str, Constraint]: ...
    @property
    def clp_relations(self) -> dict[str, Relation]: ...
    @property
    def dataset(self) -> dict[str, DatasetModel]: ...
    @property
    def weights(self) -> dict[str, Weight]: ...
