"""Glotaran Model Package

This package contains the Glotaran's base model object, the model decorators and
common model items.
"""

from glotaran.model.attribute import model_attribute
from glotaran.model.attribute import model_attribute_typed
from glotaran.model.base_model import Model
from glotaran.model.dataset_descriptor import DatasetDescriptor
from glotaran.model.decorator import model
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.util import ModelError
from glotaran.model.weight import Weight
from glotaran.plugin_system.model_registration import get_model
from glotaran.plugin_system.model_registration import is_known_model
from glotaran.plugin_system.model_registration import known_model_names
from glotaran.plugin_system.model_registration import model_plugin_table
from glotaran.plugin_system.model_registration import set_model_plugin
