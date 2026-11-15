from __future__ import annotations

from glotaran.model.item import Attribute
from glotaran.model.item import ParameterType
from glotaran.model.item import TypedItem


class Activation(TypedItem):
    compartments: dict[str, ParameterType] = Attribute(
        description="The excited compartments with their input, by compartment label."
    )
    not_normalized_compartments: list[str] = Attribute(
        factory=list,
        description=(
            "Compartments whose input is used as given. "
            "The inputs of the other compartments are divided by their sum."
        ),
    )
