"""Glotaran Model Package

This package contains the Glotaran's base model object, the model decorators and
common model items.
"""

from . import (
    dataset_descriptor,
    model,
    model_decorator,
    model_attribute,
)

# Dataset

DatasetDescriptor = dataset_descriptor.DatasetDescriptor

# BaseModel

Model = model.Model

# Decorators

model = model_decorator.model
model_attribute_typed = model_attribute.model_attribute_typed
model_attribute = model_attribute.model_attribute
