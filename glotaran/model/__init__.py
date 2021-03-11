"""Glotaran Model Package

This package contains the Glotaran's base model object, the model decorators and
common model items.
"""

from glotaran.plugin_system.model_registration import get_model
from glotaran.plugin_system.model_registration import known_model
from glotaran.plugin_system.model_registration import known_model_names

from .attribute import model_attribute
from .attribute import model_attribute_typed
from .base_model import Model
from .dataset_descriptor import DatasetDescriptor
from .decorator import model
from .weight import Weight
