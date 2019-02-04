"""Glotaran Model Package

This package contains the Glotaran's base model object, the model decorators and
common model items.
"""

from . import (
    dataset_descriptor,
    model,
    model_decorator,
    model_item,
)

# Dataset

DatasetDescriptor = dataset_descriptor.DatasetDescriptor

# BaseModel

Model = model.Model

# Decorators

model = model_decorator.model
model_item_typed = model_item.model_item_typed
model_item = model_item.model_item
