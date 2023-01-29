from collections import ChainMap
from typing import Literal
from warnings import warn

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares

from glotaran.model import ExperimentModel
from glotaran.optimization.objective import OptimizationObjectiveData
from glotaran.optimization.objective import OptimizationObjectiveExperiment
from glotaran.optimization.result import OptimizationResult
from glotaran.parameter import ParameterHistory
from glotaran.parameter import Parameters
from glotaran.utils.tee import TeeContext


class Optimization:
    def __init__(
        self,
        models: list[ExperimentModel],
        parameters: Parameters,
        verbose: bool = True,
        raise_exception: bool = False,
        maximum_number_function_evaluations: int | None = None,
        add_svd: bool = True,
        ftol: float = 1e-8,
        gtol: float = 1e-8,
        xtol: float = 1e-8,
        optimization_method: Literal[
            "TrustRegionReflection",
            "Dogbox",
            "Levenberg-Marquardt",
        ] = "TrustRegionReflection",
    ):
        self._objectives = [
            OptimizationObjectiveExperiment(e)
            if len(e.datasets) > 0
            else OptimizationObjectiveData(
                next(e.datasets.values()), e.clp_constraints, e.clp_relations, e.clp_penalties
            )
            for e in models
        ]
        self._tee = TeeContext()
        self._verbose = verbose
        self._raise = raise_exception

        self._maximum_number_function_evaluations = maximum_number_function_evaluations
        self._add_svd = add_svd
        self._ftol = ftol
        self._gtol = gtol
        self._xtol = xtol
        self._optimization_method = optimization_method

        self._parameters = parameters
        self._parameter_history = ParameterHistory()
        self._parameter_history.append(parameters)
        self._free_parameter_labels, _, _, _ = parameters.get_label_value_and_bounds_arrays(
            exclude_non_vary=True
        )

    def run(self):
        """Perform the optimization.

        Raises
        ------
        Exception
            Raised if an exception occurs during optimization and raise_exception is `True`.
        """
        (
            _,
            initial_parameter,
            lower_bounds,
            upper_bounds,
        ) = self._parameters.get_label_value_and_bounds_arrays(exclude_non_vary=True)
        ls_result = None
        termination_reason = ""
        with self._tee:
            try:
                verbose = 2 if self._verbose else 0
                ls_result = least_squares(
                    self.objective_function,
                    initial_parameter,
                    bounds=(lower_bounds, upper_bounds),
                    method=self._optimization_method,
                    max_nfev=self._maximum_number_function_evaluations,
                    verbose=verbose,
                    ftol=self._ftol,
                    gtol=self._gtol,
                    xtol=self._xtol,
                )
                termination_reason = ls_result.message
            except Exception as e:
                if self._raise:
                    raise e
                warn(f"Optimization failed:\n\n{e}")
                termination_reason = str(e)

        penalty = np.concatenate([o.calculate() for o in self._objectives])
        data = dict(ChainMap(*[o.get_result() for o in self._objectives]))
        nr_clp = len({c for d in data.values() for c in d.clp_label})
        result = OptimizationResult(
            ls_result,
            self._parameter_history,
            self._optimization_history,
            penalty,
            self._free_parameter_labels,
            termination_reason,
            nr_clp,
        )
        return self._parameters, data, result

    def objective_function(self, parameters: np.typing.ArrayLike) -> np.typing.ArrayLike:
        """Calculate the objective for the optimization.

        Parameters
        ----------
        parameters : np.typing.ArrayLike
            the parameters provided by the optimizer.

        Returns
        -------
        np.typing.ArrayLike
            The objective for the optimizer.
        """
        self._parameters.set_from_label_and_value_arrays(self._free_parameter_labels, parameters)
        return np.concatenate([o.calculate() for o in self._objectives])
