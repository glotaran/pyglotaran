from __future__ import annotations

from typing import Literal

from glotaran.builtin.items.activation.activation import Activation


class InstantActivation(Activation):
    """Excitation at the start of the model dimension, without an instrument response function."""

    type: Literal["instant"]  # type:ignore[assignment]
