"""This package contains compartment constraint items."""

from typing import List, Tuple
from glotaran.model.model_item import model_item, model_item_typed


@model_item(
    attributes={
        'compartment': str,
        'interval': List[Tuple[any, any]],
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
        return not any(interval[0] <= index <= interval[1] for interval in self.interval)


@model_item(
    attributes={
        'compartment': str,
        'interval': List[Tuple[any, any]],
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
        return any(interval[0] <= index <= interval[1] for interval in self.interval)


@model_item(attributes={
    'target': str,
    'parameter': str,
    'weight': str,
}, has_type=True, no_label=True)
class EqualAreaConstraint(ZeroConstraint):
    """An equal area constraint adds a the differenc of the sum of a
    compartements in the e matrix in one ore more intervals to the scaled sum
    of the e matrix of one or more target compartmants to resiudal. The additional
    residual is scaled with the weight.

    Parameters
    ----------
    compartment: label of the compartment
    intervals: list of tuples representing intervals on the estimated axies
    targets: list of target compartments
    parameters: list of scaling parameter for the targets
    weight: scaling factor for the residual
    """
    pass


@model_item_typed(types={
    'only': OnlyConstraint,
    'zero': ZeroConstraint,
    'equal_area': EqualAreaConstraint,
}, no_label=True)
class SpectralConstraint:
    """A compartment constraint is applied on one compartment on one or many
    intervals on the estimated axies type.

    There are three types: zeroe, equal and eqal area. See the documention of
    the respective classes for details.
    """
    pass
