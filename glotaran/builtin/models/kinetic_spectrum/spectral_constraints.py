"""This package contains compartment constraint items."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import List
from typing import Tuple

from glotaran.model import model_attribute
from glotaran.model import model_attribute_typed

if TYPE_CHECKING:
    from typing import Any

    import numpy as np

    from .kinetic_spectrum_model import KineticSpectrumModel


@model_attribute(
    properties={
        "compartment": str,
        "interval": List[Tuple[float, float]],
    },
    has_type=True,
    no_label=True,
)
class OnlyConstraint:
    """A only constraint sets the calculated matrix row of a compartment to 0
    outside the given intervals."""

    def applies(self, index: Any) -> bool:
        """
        Returns true if the index is in one of the intervals.

        Parameters
        ----------
        index :

        Returns
        -------
        applies : bool

        """

        def applies(interval):
            return interval[0] <= index <= interval[1]

        if isinstance(self.interval, tuple):
            return applies(self.interval)
        return not any([applies(i) for i in self.interval])


@model_attribute(
    properties={
        "compartment": str,
        "interval": List[Tuple[float, float]],
    },
    has_type=True,
    no_label=True,
)
class ZeroConstraint:
    """A zero constraint sets the calculated matrix row of a compartment to 0
    in the given intervals."""

    def applies(self, index: Any) -> bool:
        """
        Returns true if the indexx is in one of the intervals.

        Parameters
        ----------
        index :

        Returns
        -------
        applies : bool

        """

        def applies(interval):
            return interval[0] <= index <= interval[1]

        if isinstance(self.interval, tuple):
            return applies(self.interval)
        return any([applies(i) for i in self.interval])


@model_attribute_typed(
    types={
        "only": OnlyConstraint,
        "zero": ZeroConstraint,
    },
    no_label=True,
)
class SpectralConstraint:
    """A compartment constraint is applied on one compartment on one or many
    intervals on the estimated axis type.

    There are three types: zero, equal and equal area. See the documentation of
    the respective classes for details.
    """

    pass


def apply_spectral_constraints(
    model: KineticSpectrumModel, clp_labels: list[str], matrix: np.ndarray, index: float
) -> tuple[list[str], np.ndarray]:
    for constraint in model.spectral_constraints:
        if isinstance(constraint, (OnlyConstraint, ZeroConstraint)) and constraint.applies(index):
            idx = [not label == constraint.compartment for label in clp_labels]
            clp_labels = [label for label in clp_labels if label != constraint.compartment]
            matrix = matrix[:, idx]
    return (clp_labels, matrix)
