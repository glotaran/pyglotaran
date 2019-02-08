"""This package contains the FitResult object"""

import typing

import numpy as np
import xarray as xr
import lmfit

import glotaran  # noqa F01
from glotaran.parameter import ParameterGroup


from .grouping import create_group, create_data_group
from .optimize import calculate_residual


class Result:

    def __init__(self,
                 model: typing.Type["glotaran.model.Model"],
                 data: typing.Dict[str, typing.Union[xr.DataArray, xr.Dataset]],
                 initital_parameter: ParameterGroup,
                 nnls: bool,
                 atol: float = 0,
                 ):
        """The result of a fit.

        Parameters
        ----------
        model :
            A subclass of :class:`glotaran.model.Model`
        data :
            A dictonary containing all datasets with their labels as keys.
        initital_parameter : glotaran.parameter.ParameterGroup
            The initital fit parameter,
        nnls :
            (default = False)
            If `True` non-linear least squaes optimizing is used instead of variable projection.
        atol :
            (default = 0)
            The tolerance for grouping datasets along the estimated axis.
        """
        self._model = model
        self._data = {}
        for label, dataset in data.items():
            if model.calculated_axis not in dataset.dims:
                raise Exception("Missing coordinates for dimension "
                                f"'{model.calculated_axis}' in data for dataset "
                                f"'{label}'")
            if model.estimated_axis not in dataset.dims:
                raise Exception("Missing coordinates for dimension "
                                f"'{model.estimated_axis}' in data for dataset "
                                f"'{label}'")
            if isinstance(dataset, xr.DataArray):
                dataset = dataset.to_dataset(name="data")

            if 'weight' in dataset and 'weighted_data' not in dataset:
                dataset['weighted_data'] = np.multiply(dataset.data, dataset.weight)
            self._data[label] = dataset.transpose(model.calculated_axis, model.estimated_axis,
                                                  *[dim for dim in dataset.dims
                                                    if dim is not model.calculated_axis and
                                                    dim is not model.estimated_axis])
        self._initial_parameter = initital_parameter
        self._nnls = nnls
        self._group = create_group(model, self._data, atol)
        self._data_group = create_data_group(model, self._group, self._data)
        self._lm_result = None
        self._global_clp = {}

    @classmethod
    def from_parameter(cls,
                       model: typing.Type["glotaran.model.Model"],
                       data: typing.Dict[str, typing.Union[xr.DataArray, xr.Dataset]],
                       parameter: ParameterGroup,
                       nnls: bool,
                       atol: float = 0,
                       ) -> 'Result':
        """Creates a :attr:`FitResult` from parameters without optimization.

        Parameters
        ----------
        model :
            A subclass of :class:`glotaran.model.Model`
        data : dict(str, union(xr.Dataset, xr.DataArray))
            A dictonary containing all datasets with their labels as keys.
        parameter :
            The parameter,
        nnls :
            (default = False)
            If `True` non-linear least squaes optimizing is used instead of variable projection.
        atol :
            (default = 0)
            The tolerance for grouping datasets along the estimated axis.

        Returns
        -------
        result : FitResult
        """
        cls = cls(model, data, parameter, nnls, atol=atol)
        calculate_residual(parameter, cls)
        cls.finalize()
        return cls

    @property
    def model(self) -> typing.Type['glotaran.model.Model']:
        """The model for analysis."""
        return self._model

    @property
    def nnls(self) -> bool:
        """If `True` non-linear least squaes optimizing is used instead of variable
        projection."""
        return self._nnls

    @property
    def data(self) -> typing.Dict[str, xr.Dataset]:
        """The resulting data as a dictionary of `xarray.Dataset`.

        Note
        ----
        The actual content of the data depends on the actual model and can be found in the
        documentation for the model.
        """
        return self._data

    @property
    def nfev(self) -> int:
        """The number of function evaluations."""
        return self._lm_result.nfev if self._lm_result else 0

    @property
    def nvars(self) -> int:
        """Number of variables in optimization."""
        return self._lm_result.nvarys if self._lm_result else None

    @property
    def ndata(self) -> int:
        """Number of data points."""
        return self._lm_result.ndata if self._lm_result else None

    @property
    def nfree(self) -> int:
        """Degrees of freedom in optimization. """
        return self._lm_result.nfree if self._lm_result else None

    @property
    def chisqr(self) -> float:
        """The chi-square of the optimization """
        return self._lm_result.chisqr if self._lm_result else 0

    @property
    def red_chisqr(self) -> float:
        """The reduced chi-square of the optimization."""
        return self._lm_result.redchi if self._lm_result else 0

    @property
    def root_mean_sqare_error(self) -> float:
        """The root mean square error the optimization."""
        return np.sqrt(self.red_chisqr)

    @property
    def var_names(self) -> typing.List[str]:
        """Ordered list of variable parameter names used in optimization, and
        useful for understanding the values in :attr:`covar`."""
        return [n.replace('_', '.') for n in self._lm_result.var_names] \
            if self._lm_result else None

    @property
    def covar(self) -> np.ndarray:
        """Covariance matrix from minimization, with rows and columns
        corresponding to :attr:`var_names`."""
        return self._lm_result.covar if self._lm_result else None

    @property
    def best_fit_parameter(self) -> ParameterGroup:
        """The best fit parameters."""
        if self._lm_result is None:
            return self.initial_parameter
        return ParameterGroup.from_parameter_dict(self._lm_result.params)

    @property
    def initial_parameter(self) -> ParameterGroup:
        """The initital fit parameter"""
        return self._initial_parameter

    @property
    def global_clp(self) -> typing.Dict[any, xr.DataArray]:
        """A dictonary of the global condionally linear parameter with the index on the global
        estimated axis as keys."""
        return self._global_clp

    @property
    def data_groups(self) -> typing.Dict[any, np.ndarray]:
        """A dictonary of the data groups along the estimated axis."""
        return self._data_group

    @property
    def groups(self) -> typing.Dict[any, typing.List[typing.Tuple[any, str]]]:
        """A dictonary of the dataset_descriptor groups along the estimated axis."""
        return self._group

    def finalize(self, lm_result: lmfit.minimizer.MinimizerResult = None):

        if lm_result:
            self._lm_result = lm_result

        for label in self.model.dataset:
            dataset = self._data[label]

            if 'weight' in dataset:
                dataset['weighted_residual'] = dataset.residual
                dataset.residual = np.multiply(dataset.weighted_residual, dataset.weight**-1)

            l, v, r = np.linalg.svd(dataset.residual)

            dataset['residual_left_singular_vectors'] = \
                ((self.model.calculated_axis, 'left_singular_value_index'), l)

            dataset['residual_right_singular_vectors'] = \
                (('right_singular_value_index', self.model.estimated_axis), r)

            dataset['residual_singular_values'] = \
                ((self.model.estimated_axis, 'singular_value_index'), r)

            # reconstruct fitted data

            dataset['fitted_data'] = dataset.data - dataset.residual

        if callable(self.model._finalize_result):
            self.model._finalize_result(self)

    def mprint(self, with_model=True):
        string = "# Fitresult\n\n"

        ll = 32
        lr = 13

        string += "Optimization Result".ljust(ll-1)
        string += "|"
        string += "|".rjust(lr)
        string += "\n"
        string += "|".rjust(ll, "-")
        string += "|".rjust(lr, "-")
        string += "\n"

        string += "Number of residual evaluation |".rjust(ll)
        string += f"{self.nfev} |".rjust(lr)
        string += "\n"
        string += "Number of variables |".rjust(ll)
        string += f"{self.nvars} |".rjust(lr)
        string += "\n"
        string += "Number of datapoints |".rjust(ll)
        string += f"{self.ndata} |".rjust(lr)
        string += "\n"
        string += "Negrees of freedom |".rjust(ll)
        string += f"{self.nfree} |".rjust(lr)
        string += "\n"
        string += "Chi Square |".rjust(ll)
        string += f"{self.chisqr:.2e} |".rjust(lr)
        string += "\n"
        string += "Reduced Chi Square |".rjust(ll)
        string += f"{self.red_chisqr:.2e} |".rjust(lr)
        string += "\n"
        string += "Root Mean Square Error |".rjust(ll)
        string += f"{self.root_mean_sqare_error:.2e} |".rjust(lr)
        string += "\n"
        #
        #  string += "\n"
        #  string += "## Best Fit Parameter\n\n"
        #  string += f"{self.best_fit_parameter}"
        #  string += "\n"

        if with_model:

            string += "\n\n" + self.model.mprint(parameter=self.best_fit_parameter,
                                                 initial=self.initial_parameter)

        return string

    def __str__(self):
        return self.mprint(with_model=False)
