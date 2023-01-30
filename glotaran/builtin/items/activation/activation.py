from glotaran.model import Attribute
from glotaran.model import LibraryItemTyped
from glotaran.model import ParameterType


class Activation(LibraryItemTyped):
    compartments: dict[str, ParameterType] = Attribute(
        description="A dictionary of the activated compartments with the activation aamplitude."
    )
