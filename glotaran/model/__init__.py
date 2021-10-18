"""Glotaran Model Package

This package contains the Glotaran's base model object, the model decorators and
common model items.
"""

from glotaran.model.clp_penalties import EqualAreaPenalty
from glotaran.model.constraint import Constraint
from glotaran.model.constraint import OnlyConstraint
from glotaran.model.constraint import ZeroConstraint
from glotaran.model.dataset_group import DatasetGroup
from glotaran.model.dataset_group import DatasetGroupModel
from glotaran.model.dataset_model import DatasetModel
from glotaran.model.item import model_item
from glotaran.model.item import model_item_typed
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.megacomplex import megacomplex
from glotaran.model.model import Model
from glotaran.model.relation import Relation
from glotaran.model.util import ModelError
from glotaran.model.weight import Weight
from glotaran.plugin_system.megacomplex_registration import get_megacomplex
from glotaran.plugin_system.megacomplex_registration import is_known_megacomplex
from glotaran.plugin_system.megacomplex_registration import known_megacomplex_names
from glotaran.plugin_system.megacomplex_registration import megacomplex_plugin_table
from glotaran.plugin_system.megacomplex_registration import set_megacomplex_plugin
