"""This package contains compartment constraint items."""

import typing
import numpy as np

from glotaran.model import model_attribute, model_attribute_typed


@model_attribute(
    properties={
        'compartment': str,
        'interval': typing.List[typing.Tuple[float, float]],
    }, has_type=True, no_label=True)
class OnlyConstraint:
    """A only constraint sets the calculated matrix row of a compartment to 0
    outside the given intervals. """

    def applies(self, index: any) -> bool:
        """
        Returns true if the indexx is in one of the intervals.

        Parameters
        ----------
        index : any

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
        'compartment': str,
        'interval': typing.List[typing.Tuple[float, float]],
    }, has_type=True, no_label=True)
class ZeroConstraint:
    """A zero constraint sets the calculated matrix row of a compartment to 0
    in the given intervals. """

    def applies(self, index: any) -> bool:
        """
        Returns true if the indexx is in one of the intervals.

        Parameters
        ----------
        index : any

        Returns
        -------
        applies : bool

        """
        def applies(interval):
            return interval[0] <= index <= interval[1]
        if isinstance(self.interval, tuple):
            return applies(self.interval)
        return any([applies(i) for i in self.interval])


@model_attribute_typed(types={
    'only': OnlyConstraint,
    'zero': ZeroConstraint,
}, no_label=True)
class SpectralConstraint:
    """A compartment constraint is applied on one compartment on one or many
    intervals on the estimated axies type.

    There are three types: zeroe, equal and eqal area. See the documention of
    the respective classes for details.
    """
    pass


def apply_spectral_constraints(
        model: typing.Type['KineticModel'],
        clp_labels: typing.List[str],
        matrix: np.ndarray,
        index: float) -> typing.Tuple[typing.List[str], np.ndarray]:
    for constraint in model.spectral_constraints:
        if isinstance(constraint, (OnlyConstraint, ZeroConstraint)) and constraint.applies(index):
            idx = [not label == constraint.compartment for label in clp_labels]
            clp_labels = [label for label in clp_labels if not label == constraint.compartment]
            matrix = matrix[:, idx]
    return (clp_labels, matrix)
