"""Module containing the optimize function."""
from __future__ import annotations

from glotaran.optimization.optimizer import Optimizer
from glotaran.project import Result
from glotaran.project import Scheme


def optimize(scheme: Scheme, verbose: bool = True, raise_exception: bool = False) -> Result:
    """Optimize a scheme.

    Parameters
    ----------
    scheme : Scheme
        The optimization scheme.
    verbose : bool
        Deactivate printing of logs if `False`.
    raise_exception : bool
        Raise exceptions during optimizations instead of gracefully exiting if `True`.

    Returns
    -------
    Result
        The result of the optimization.
    """
    optimizer = Optimizer(scheme, verbose, raise_exception)
    optimizer.optimize()
    return optimizer.create_result()
