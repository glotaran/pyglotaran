""" Glotaran Spectral Relation """

import typing
import numpy as np

from glotaran.model import model_item


@model_item(
    attributes={
        'compartment': str,
        'target': str,
        'parameter': str,
        'interval': typing.List[typing.Tuple[any, any]],
    }, no_label=True)
class SpectralRelation:
    def applies(self, index: any) -> bool:
        """
        Returns true if the index is in one of the intervals.

        Parameters
        ----------
        index : any

        Returns
        -------
        applies : bool

        """
        return any(interval[0] <= index <= interval[1] for interval in self.interval)
