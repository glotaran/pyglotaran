from typing import List, Tuple
from .model_item import glotaran_model_item, glotaran_model_item_typed


@glotaran_model_item(attributes={
                        'compartment': str,
                        'interval': List[Tuple[any, any]],
                    }, has_type=True, no_label=True)
class ZeroConstraint:
    """A zero constraint sets the c matrix of a compartment to 0.
    """
    def applies(self, x):
        """
        Returns true if x is in one of the intervals.
        Parameters
        ----------
        x : point on the estimated axies

        """
        return any(interval[0] <= x <= interval[1] for interval in self.intervals)


@glotaran_model_item(attributes={
                        'targets': List[str],
                        'parameters': List[str],
                    }, has_type=True)
class EqualConstraint(ZeroConstraint):
    """An equal constraint The compartments c matrix will be replaced with a sum
    of target compartments, each scaled by a parameter.

    C = p1 * C_t1 + p2 * C_t1 + ...

    Parameters
    ----------
    compartment: label of the compartment
    intervals: list of tuples representing intervals on the estimated axies
    targets: list of target compartments
    parameters: list of scaling parameter for the targets
    """
    def parameter_and_targets(self):
        """generates traget and parameter pairs """
        for i in range(len(self.parameters)):
            yield self.parameters[i], self.targets[i]


@glotaran_model_item(attributes={
                        'weight': str,
                    }, has_type=True)
class EqualAreaConstraint(EqualConstraint):
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


@glotaran_model_item_typed(types={
    'zero': ZeroConstraint,
    'equal': EqualConstraint,
    'equal_area': EqualAreaConstraint,
}
)
class CompartmentConstraint:
    """A compartment constraint is applied on one compartment on one or many
    intervals on the estimated axies type.
    """
    pass
