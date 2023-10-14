from typing import Literal

from glotaran.builtin.items.activation.activation import Activation


class InstantActivation(Activation):
    type: Literal["instant"]  # type:ignore[assignment]
