from __future__ import annotations

from glotaran.optimization.optimizer import Optimizer
from glotaran.project import Result
from glotaran.project import Scheme


def optimize(scheme: Scheme, verbose: bool = True, raise_exception: bool = False) -> Result:

    optimizer = Optimizer(scheme, verbose, raise_exception)
    optimizer.optimize()
    return optimizer.create_result()
