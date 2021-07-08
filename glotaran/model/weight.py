"""The Weight property class."""

from typing import List
from typing import Tuple

from glotaran.model.item import model_item


@model_item(
    properties={
        "datasets": {type: List[str]},
        "global_interval": {
            "type": List[Tuple[float, float]],
            "default": None,
            "allow_none": True,
        },
        "model_interval": {
            "type": List[Tuple[float, float]],
            "default": None,
            "allow_none": True,
        },
        "value": {"type": float},
    },
    has_label=False,
)
class Weight:
    """The `Weight` class describes a value by which a dataset will scaled.

    `global_interval` and `model_interval` are optional. The whole range of the dataset
    will be used if not set.
    """
