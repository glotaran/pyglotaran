"""Module containing the optimizer class."""
from __future__ import annotations

from warnings import warn

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares

from glotaran import __version__ as glotaran_version
from glotaran.optimization.optimization_group import OptimizationGroup
from glotaran.parameter import ParameterHistory
from glotaran.project import Result
from glotaran.project import Scheme

SUPPORTED_METHODS = {
    "TrustRegionReflection": "trf",
    "Dogbox": "dogbox",
    "Levenberg-Marquardt": "lm",
}


class InitialParameterError(ValueError):
    """Inidcates that initial parameters can not be evaluated."""

    def __init__(self):
        """Initialize a InitialParameterError."""
        super().__init__("Initial parameters can not be evaluated.")


class ParameterNotInitializedError(ValueError):
    """Inidcates that scheme parameters are not initialized."""

    def __init__(self):
        """Initialize a ParameterNotInitializedError."""
        super().__init__("Parameter not initialized")


class MissingDatasetsError(ValueError):
    """Inidcates that datasets are missing in the scheme."""

    def __init__(self, missing_datasets: list[str]):
        """Initialize a MissingDatasetsError.

        Parameters
        ----------
        missing_datasets : list[str]
            The missing datasets.
        """
        super().__init__(f"Missing data for datasets: {missing_datasets}")


class UnsupportedMethodError(ValueError):
    """Inidcates that the optimization method is unsupported."""

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
        return self.calculate_penalty()

    def calculate_penalty(self) -> np.typing.ArrayLike:
        """Calculate the penalty of the scheme.

        Returns
        -------
        np.typing.ArrayLike
            The penalty.
        """
        for group in self._optimization_groups:
            group.calculate(self._parameters)
        self._parameter_history.append(self._parameters)

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
        #  result: OptimizeResult = self._optimization_result

        number_of_function_evaluation = (
            self._optimization_result.nfev
            if success
            else self._parameter_history.number_of_records
        )
        number_of_jacobian_evaluation = self._optimization_result.njev if success else None
        optimality = float(self._optimization_result.optimality) if success else None
        number_of_data_points = self._optimization_result.fun.size if success else None
        number_of_parameters = self._optimization_result.x.size if success else None
        degrees_of_freedom = number_of_data_points - number_of_parameters if success else None
        chi_square = float(np.sum(self._optimization_result.fun**2)) if success else None
        reduced_chi_square = chi_square / degrees_of_freedom if success else None
        root_mean_square_error = float(np.sqrt(reduced_chi_square)) if success else None
        jacobian = self._optimization_result.jac if success else None

        if success:
            self._parameters.set_from_label_and_value_arrays(
                self._free_parameter_labels, self._optimization_result.x
            )
        elif self._parameter_history.number_of_records == 1:
            raise InitialParameterError()
        else:
            self._parameters.set_from_history(self._parameter_history, -2)

        full_penalty = self.calculate_penalty()
        data = {}
        for group in self._optimization_groups:
            group.calculate(self._parameters)
            data.update(group.create_result_data())

        covariance_matrix = None
        if success:
            # See PR #706: More robust covariance matrix calculation
            _, jacobian_SV, jacobian_RSV = np.linalg.svd(jacobian, full_matrices=False)
            jacobian_SV_square = jacobian_SV**2
            mask = jacobian_SV_square > np.finfo(float).eps
            covariance_matrix = (jacobian_RSV[mask].T / jacobian_SV_square[mask]) @ jacobian_RSV[
                mask
            ]
            standard_errors = root_mean_square_error * np.sqrt(np.diag(covariance_matrix))
            for label, error in zip(self._free_parameter_labels, standard_errors):
                self._parameters.get(label).standard_error = error

        cost = 0.5 * np.dot(full_penalty, full_penalty)

        return Result(
            cost=cost,
            data=data,
            glotaran_version=glotaran_version,
            free_parameter_labels=self._free_parameter_labels,
            number_of_function_evaluations=number_of_function_evaluation,
            initial_parameters=self._scheme.parameters,
            optimized_parameters=self._parameters,
            parameter_history=self._parameter_history,
            scheme=self._scheme,
            success=success,
            termination_reason=self._termination_reason,
            chi_square=chi_square,
            covariance_matrix=covariance_matrix,
            degrees_of_freedom=degrees_of_freedom,
            jacobian=jacobian,
            number_of_data_points=number_of_data_points,
            number_of_jacobian_evaluations=number_of_jacobian_evaluation,
            number_of_parameters=number_of_parameters,
            optimality=optimality,
            reduced_chi_square=reduced_chi_square,
            root_mean_square_error=root_mean_square_error,
        )
