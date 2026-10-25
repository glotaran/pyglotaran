from __future__ import annotations

from glotaran.model.item import Attribute
from glotaran.model.item import ParameterType
from glotaran.model.item import TypedItem


class Activation(TypedItem):
    compartments: dict[str, ParameterType] = Attribute(
        description="A dictionary of activated compartments with the activation amplitude."
    )
    not_normalized_compartments: list[str] = Attribute(
        factory=list,
        description="A list of the compartments which will not be normalized.",
    )
