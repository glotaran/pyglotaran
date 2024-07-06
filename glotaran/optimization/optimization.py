from __future__ import annotations

from collections import ChainMap
from typing import TYPE_CHECKING
from typing import Literal
from warnings import warn

import numpy as np
from scipy.optimize import least_squares

from glotaran.model.errors import GlotaranModelIssues
from glotaran.model.errors import GlotaranUserError
from glotaran.optimization.objective import OptimizationObjective
from glotaran.optimization.optimization_history import OptimizationHistory
from glotaran.optimization.result import OptimizationResult
from glotaran.parameter import ParameterHistory
from glotaran.parameter import Parameters
from glotaran.utils.tee import TeeContext

if TYPE_CHECKING:
    import xarray as xr

    from glotaran.model.experiment_model import ExperimentModel
    from glotaran.project.library import ModelLibrary
    from glotaran.typing.types import ArrayLike


SUPPORTED_OPTIMIZATION_METHODS = {
    "TrustRegionReflection": "trf",
    "Dogbox": "dogbox",
    "Levenberg-Marquardt": "lm",
}


class UnsupportedMethodError(GlotaranUserError):
    """Indicates that the optimization method is unsupported."""

    def __init__(self, method: str):
        """Initialize an UnsupportedMethodError.

        Parameters
        ----------
        method : str
            The unsupported method.
        """
        super().__init__(
            f"Unsupported optimization method {method}. "
            f"Supported methods are '{list(SUPPORTED_OPTIMIZATION_METHODS.keys())}'"
        )


class Optimization:
    def __init__(
        self,
        models: list[ExperimentModel],
        parameters: Parameters,
        library: ModelLibrary,
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
        self._parameters = Parameters.empty()
        models = [
            experiment.resolve(library, self._parameters, initial=parameters)
            for experiment in models
        ]
        issues = [issue for experiment in models for issue in experiment.get_issues(parameters)]
        if len(issues) > 0:
            raise GlotaranModelIssues(issues)
        self._objectives = [OptimizationObjective(experiment) for experiment in models]
        self._tee = TeeContext()
        self._verbose = verbose
        self._raise = raise_exception

        self._maximum_number_function_evaluations = maximum_number_function_evaluations
        self._add_svd = add_svd
        self._ftol = ftol
        self._gtol = gtol
        self._xtol = xtol
        if optimization_method not in SUPPORTED_OPTIMIZATION_METHODS:
            raise UnsupportedMethodError(optimization_method)
        self._optimization_method = SUPPORTED_OPTIMIZATION_METHODS[optimization_method]

        self._parameter_history = ParameterHistory()
        self._parameter_history.append(self._parameters)
        self._free_parameter_labels, _, _, _ = self._parameters.get_label_value_and_bounds_arrays(
            exclude_non_vary=True
        )

    def run(self) -> tuple[Parameters, dict[str, xr.Dataset], OptimizationResult]:
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
            # No matter the error we want to behave gracefully
            except Exception as e:  # noqa: BLE001
                if self._raise:
                    raise e
                warn(f"Optimization failed:\n\n{e}")
                termination_reason = str(e)

        penalty = np.concatenate([o.calculate() for o in self._objectives])
        results = [o.get_result() for o in self._objectives]
        data = dict(ChainMap(*[r.data for r in results]))
        nr_clp = sum(r.free_clp_size for r in results)
        result = OptimizationResult.from_least_squares_result(
            ls_result,
            self._parameter_history,
            OptimizationHistory.from_stdout_str(self._tee.read()),
            penalty,
            self._free_parameter_labels,
            termination_reason,
            nr_clp,
        )
        return self._parameters, data, result

    def dry_run(self) -> tuple[Parameters, dict[str, xr.Dataset], OptimizationResult]:
        termination_reason = "Dry run."

        penalty = np.concatenate([o.calculate() for o in self._objectives])
        results = [o.get_result() for o in self._objectives]
        data = dict(ChainMap(*[r.data for r in results]))
        nr_clp = sum(r.free_clp_size for r in results)
        result = OptimizationResult.from_least_squares_result(
            None,
            self._parameter_history,
            OptimizationHistory.from_stdout_str(self._tee.read()),
            penalty,
            self._free_parameter_labels,
            termination_reason,
            nr_clp,
        )
        return self._parameters, data, result

    def objective_function(self, parameters: ArrayLike) -> ArrayLike:
        """Calculate the objective for the optimization.

        Parameters
        ----------
        parameters : ArrayLike
            the parameters provided by the optimizer.

        Returns
        -------
        ArrayLike
            The objective for the optimizer.
        """
        self._parameters.set_from_label_and_value_arrays(self._free_parameter_labels, parameters)
        return np.concatenate([o.calculate() for o in self._objectives])
