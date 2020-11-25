"""Glotaran Model Package

This package contains the Glotaran's base model object, the model decorators and
common model items.
"""

from .attribute import model_attribute  # noqa: F401
from .attribute import model_attribute_typed  # noqa: F401
from .base_model import Model  # noqa: F401
from .dataset_descriptor import DatasetDescriptor  # noqa: F401
from .decorator import model  # noqa: F401
from .weight import Weight  # noqa: F401
