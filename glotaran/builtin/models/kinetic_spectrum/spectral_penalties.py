"""This package contains compartment constraint items."""

import typing

from glotaran.model import model_attribute
from glotaran.parameter import Parameter


@model_attribute(properties={
    'compartment': str,
    'interval': typing.List[typing.Tuple[float, float]],
    'target': str,
    'parameter': Parameter,
    'weight': str,
}, no_label=True)
class EqualAreaPenalty:
    """An equal area constraint adds a the differenc of the sum of a
    compartements in the e matrix in one ore more intervals to the scaled sum
    of the e matrix of one or more target compartmants to resiudal. The additional
    residual is scaled with the weight."""

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
