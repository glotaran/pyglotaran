"""The gloataran model module."""
from glotaran.model.clp_constraint import OnlyConstraint
from glotaran.model.clp_constraint import ZeroConstraint
from glotaran.model.clp_penalties import EqualAreaPenalty
from glotaran.model.clp_relation import Relation
from glotaran.model.dataset_group import DatasetGroup
from glotaran.model.dataset_model import DatasetModel
from glotaran.model.dataset_model import get_dataset_model_model_dimension
from glotaran.model.dataset_model import is_dataset_model_index_dependent
from glotaran.model.item import ItemIssue
from glotaran.model.item import ModelItem
from glotaran.model.item import ModelItemType
from glotaran.model.item import ModelItemTyped
from glotaran.model.item import ParameterType
from glotaran.model.item import attribute
from glotaran.model.item import fill_item
from glotaran.model.item import item
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.megacomplex import megacomplex
from glotaran.model.model import Model
from glotaran.model.model import ModelError
from glotaran.model.weight import Weight
