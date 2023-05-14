"""Module containing the optimizer class."""
from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares

from glotaran import __version__ as glotaran_version
from glotaran.optimization.optimization_group import OptimizationGroup
from glotaran.optimization.optimization_history import OptimizationHistory
from glotaran.parameter import ParameterHistory
from glotaran.parameter.parameter import _log_value
from glotaran.project import Result
from glotaran.project import Scheme
from glotaran.utils.regex import RegexPattern
from glotaran.utils.tee import TeeContext

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike

SUPPORTED_METHODS = {
    "TrustRegionReflection": "trf",
    "Dogbox": "dogbox",
    "Levenberg-Marquardt": "lm",
}


class InitialParameterError(ValueError):
    """Indicates that initial parameters can not be evaluated."""

    def __init__(self):
        """Initialize a InitialParameterError."""
        super().__init__("Initial parameters can not be evaluated.")


class ParameterNotInitializedError(ValueError):
    """Indicates that scheme parameters are not initialized."""

    def __init__(self):
        """Initialize a ParameterNotInitializedError."""
        super().__init__("Parameter not initialized")


class MissingDatasetsError(ValueError):
    """Indicates that datasets are missing in the scheme."""

    def __init__(self, missing_datasets: list[str]):
        """Initialize a MissingDatasetsError.

        Parameters
        ----------
        missing_datasets : list[str]
            The missing datasets.
        """
        super().__init__(f"Missing data for datasets: {missing_datasets}")


class UnsupportedMethodError(ValueError):
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
            f"Supported methods are '{list(SUPPORTED_METHODS.keys())}'"
        )


class Optimizer:
    """A class to optimize a scheme."""

    def __init__(self, scheme: Scheme, verbose: bool = True, raise_exception: bool = False):
        """Initialize an optimization group for a dataset group.

        Parameters
        ----------
        scheme : Scheme
            The optimization scheme.
        verbose : bool
            Deactivate printing of logs if `False`.
        raise_exception : bool
            Raise exceptions during optimizations instead of gracefully exiting if `True`.

        Raises
        ------
        MissingDatasetsError
            Raised if datasets are missing.
        ParameterNotInitializedError
            Raised if the scheme parameters are `None`.
        UnsupportedMethodError
            Raised if the optimization method is unsupported.
        """
        if missing_datasets := [
            label for label in scheme.model.dataset if label not in scheme.data
        ]:
            raise MissingDatasetsError(missing_datasets)
        if scheme.parameters is None:
            raise ParameterNotInitializedError()
        self._parameters = scheme.parameters.copy()
        if scheme.optimization_method not in SUPPORTED_METHODS:
            raise UnsupportedMethodError(scheme.optimization_method)
        self._method = SUPPORTED_METHODS[scheme.optimization_method]

        self._scheme = scheme
        self._tee = TeeContext()
        self._verbose = verbose
        self._raise = raise_exception

        self._optimization_result: OptimizeResult = None
        self._termination_reason = ""

        self._optimization_groups = [
            OptimizationGroup(scheme, group)
            for group in scheme.model.get_dataset_groups().values()
        ]

        self._parameter_history = ParameterHistory()
        self._parameter_history.append(scheme.parameters)

    def optimize(self):
        """Perform the optimization.

        Raises
        ------
        Exception
            Raised if an exception occurs during optimization and raise_exception is `True`.
        """
        (
            self._free_parameter_labels,
            initial_parameter,
            lower_bounds,
            upper_bounds,
        ) = self._scheme.parameters.get_label_value_and_bounds_arrays(exclude_non_vary=True)
        with self._tee:
            try:
                verbose = 2 if self._verbose else 0
                self._optimization_result = least_squares(
                    self.objective_function,
                    initial_parameter,
                    bounds=(lower_bounds, upper_bounds),
                    method=self._method,
                    max_nfev=self._scheme.maximum_number_function_evaluations,
                    verbose=verbose,
                    ftol=self._scheme.ftol,
                    gtol=self._scheme.gtol,
                    xtol=self._scheme.xtol,
                )
                self._termination_reason = self._optimization_result.message
            except Exception as e:
                if self._raise:
                    raise e
                warn(f"Optimization failed:\n\n{e}")
                self._termination_reason = str(e)

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
        return self.calculate_penalty()

    def calculate_penalty(self) -> ArrayLike:
        """Calculate the penalty of the scheme.

        Returns
        -------
        ArrayLike
            The penalty.
        """
        for group in self._optimization_groups:
            group.calculate(self._parameters)
        self._parameter_history.append(
            self._parameters, self.get_current_optimization_iteration(self._tee.read())
        )

        penalties = [group.get_full_penalty() for group in self._optimization_groups]

        return np.concatenate(penalties) if len(penalties) != 1 else penalties[0]

    def create_result(self) -> Result:
        """Create the result of the optimization.

        Returns
        -------
        Result
            The result of the optimization.

        Raises
        ------
        InitialParameterError
            Raised if the initial parameters could not be evaluated.
        """
        success = self._optimization_result is not None

        if self._parameter_history.number_of_records == 1:
            raise InitialParameterError()
        elif not success:
            self._parameters.set_from_history(self._parameter_history, -2)

        result_args = {
            "success": success,
            "scheme": self._scheme,
            "glotaran_version": glotaran_version,
            "free_parameter_labels": self._free_parameter_labels,
            "initial_parameters": self._scheme.parameters,
            "parameter_history": self._parameter_history,
            "termination_reason": self._termination_reason,
            "optimization_history": OptimizationHistory.from_stdout_str(self._tee.read()),
            "number_of_function_evaluations": self._optimization_result.nfev
            if success
            else self._parameter_history.number_of_records,
        }

        if success:
            result_args["number_of_jacobian_evaluations"] = self._optimization_result.njev
            result_args["optimality"] = float(self._optimization_result.optimality)
            result_args["number_of_residuals"] = self._optimization_result.fun.size
            result_args["number_of_clps"] = sum(
                group.number_of_clps for group in self._optimization_groups
            )
            result_args["number_of_free_parameters"] = self._optimization_result.x.size
            result_args["degrees_of_freedom"] = (
                result_args["number_of_residuals"]
                - result_args["number_of_free_parameters"]
                - result_args["number_of_clps"]
            )
            result_args["chi_square"] = float(np.sum(self._optimization_result.fun**2))
            result_args["reduced_chi_square"] = (
                result_args["chi_square"] / result_args["degrees_of_freedom"]
            )
            result_args["root_mean_square_error"] = float(
                np.sqrt(result_args["reduced_chi_square"])
            )
            result_args["jacobian"] = self._optimization_result.jac

            self._parameters.set_from_label_and_value_arrays(
                self._free_parameter_labels, self._optimization_result.x
            )
            result_args[
                "covariance_matrix"
            ] = self.calculate_covariance_matrix_and_standard_errors(
                result_args["jacobian"], result_args["root_mean_square_error"]
            )

        result_args["additional_penalty"] = [
            group.get_additional_penalties() for group in self._optimization_groups
        ]

        full_penalty = self.calculate_penalty()
        result_args["cost"] = 0.5 * np.dot(full_penalty, full_penalty)

        result_args["optimized_parameters"] = self._parameters

        result_args["data"] = {}
        for group in self._optimization_groups:
            group.calculate(self._parameters)
            result_args["data"].update(group.create_result_data())

        return Result(**result_args)

    def calculate_covariance_matrix_and_standard_errors(
        self, jacobian: ArrayLike, root_mean_square_error: float
    ) -> ArrayLike:
        """Calculate the covariance matrix and standard errors of the optimization.

        Parameters
        ----------
        jacobian : ArrayLike
            The jacobian matrix.
        root_mean_square_error : float
            The root mean square error.

        Returns
        -------
        ArrayLike
            The covariance matrix.
        """
        # See PR #706: More robust covariance matrix calculation
        _, jacobian_sv, jacobian_rsv = np.linalg.svd(jacobian, full_matrices=False)
        jacobian_sv_square = jacobian_sv**2
        mask = jacobian_sv_square > np.finfo(float).eps
        covariance_matrix = (jacobian_rsv[mask].T / jacobian_sv_square[mask]) @ jacobian_rsv[mask]
        standard_errors = root_mean_square_error * np.sqrt(np.diag(covariance_matrix))
        for label, error in zip(self._free_parameter_labels, standard_errors):
            parameter = self._parameters.get(label)
            if parameter.non_negative:
                if error < np.abs(_log_value(parameter.value)):
                    parameter.standard_error = parameter.value * (np.exp(error) - 1.0)
                else:
                    parameter.standard_error = np.abs(parameter.value)
            else:
                self._parameters.get(label).standard_error = error
        return covariance_matrix

    @staticmethod
    def get_current_optimization_iteration(optimize_stdout: str) -> int:
        """Extract current iteration from ``optimize_stdout``.

        Parameters
        ----------
        optimize_stdout: str
            SciPy optimization stdout string, read out via ``TeeContext.read()``.

        Returns
        -------
        int
            Current iteration (``0`` if pattern did not match).
        """
        matches = RegexPattern.optimization_stdout.findall(optimize_stdout)
        return 0 if len(matches) == 0 else int(matches[-1][0])
