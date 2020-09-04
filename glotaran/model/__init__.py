"""Glotaran Model Package

This package contains the Glotaran's base model object, the model decorators and
common model items.
"""

from . import dataset_descriptor
from . import model
from . import model_attribute
from . import model_decorator

# Dataset

DatasetDescriptor = dataset_descriptor.DatasetDescriptor

# BaseModel

Model = model.Model

# Decorators

model = model_decorator.model
model_attribute_typed = model_attribute.model_attribute_typed
model_attribute = model_attribute.model_attribute
