"""Glotaran Model Package

This package contains the Glotaran's base model object, the model decorators and
common model items.
"""

from glotaran.model.clp_penalties import EqualAreaPenalty
from glotaran.model.constraint import Constraint
from glotaran.model.constraint import OnlyConstraint
from glotaran.model.constraint import ZeroConstraint
from glotaran.model.dataset_model import DatasetModel
from glotaran.model.item import model_item
from glotaran.model.item import model_item_typed
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.megacomplex import megacomplex
from glotaran.model.model import Model
from glotaran.model.relation import Relation
from glotaran.model.util import ModelError
from glotaran.model.weight import Weight
from glotaran.plugin_system.model_registration import get_model
from glotaran.plugin_system.model_registration import is_known_model
from glotaran.plugin_system.model_registration import known_model_names
from glotaran.plugin_system.model_registration import model_plugin_table
from glotaran.plugin_system.model_registration import set_model_plugin
