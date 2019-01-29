"""This package contains the FitResult object"""

import typing

import numpy as np
import xarray as xr
from lmfit.minimizer import Minimizer

import glotaran  # noqa F01
from glotaran.parameter import ParameterGroup


from .grouping import create_group, create_data_group
from .grouping import calculate_group_item
from .variable_projection import residual_variable_projection
from .nnls import residual_nnls


class FitResult:

    def __init__(self,
                 model: typing.Type["glotaran.model.BaseModel"],
                 data: typing.Dict[str, typing.Union[xr.DataArray, xr.Dataset]],
                 initital_parameter: ParameterGroup,
                 nnls: bool,
                 atol: float = 0,
                 ):
        """The result of a fit.

        Parameters
        ----------
        model :  glotaran.model.BaseModel
            A subclass of :class:`glotaran.model.BaseModel`
        data : dict(str, union(xr.Dataset, xr.DataArray))
            A dictonary containing all datasets with their labels as keys.
        initital_parameter : glotaran.parameter.ParameterGroup
            The initital fit parameter,
        nnls : bool, optional
            (default = False)
            If `True` non-linear least squaes optimizing is used instead of variable projection.
        atol : float, optional
            (default = 0)
            The tolerance for grouping datasets along the estimated axis.
        """
        self.model = model
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
                       model: typing.Type["glotaran.model.BaseModel"],
                       data: typing.Dict[str, typing.Union[xr.DataArray, xr.Dataset]],
                       parameter: ParameterGroup,
                       nnls: bool,
                       atol: float = 0,
                       ) -> 'FitResult':
        """Creates a :attr:`FitResult` from parameters without optimization.

        Parameters
        ----------
        model :  glotaran.model.BaseModel
            A subclass of :class:`glotaran.model.BaseModel`
        data : dict(str, union(xr.Dataset, xr.DataArray))
            A dictonary containing all datasets with their labels as keys.
        parameter : glotaran.model.ParameterGroup
            The parameter,
        nnls : bool, optional
            (default = False)
            If `True` non-linear least squaes optimizing is used instead of variable projection.
        atol : float, optional
            (default = 0)
            The tolerance for grouping datasets along the estimated axis.

        Returns
        -------
        result : FitResult
        """
        cls = cls(model, data, parameter, nnls, atol=atol)
        cls._calculate_residual(parameter)
        cls._finalize()
        return cls

    def optimize(self, verbose: bool = True, max_nfev: int = None):
        """Optimizes the parameter.

        Parameters
        ----------
        verbose : bool, optional
            (default = True)
            If `True` feedback is printed at every iteration.
        max_nfev : int, optional
            (default = None)
            Maximum number of function evaluations. `None` for unlimited.

        Returns
        -------
        """
        parameter = self.initial_parameter.as_parameter_dict()
        minimizer = Minimizer(
            self._calculate_residual,
            parameter,
            fcn_args=[],
            fcn_kws=None,
            iter_cb=self._iter_cb,
            scale_covar=True,
            nan_policy='omit',
            reduce_fcn=None,
            **{})
        verbose = 2 if verbose else 0
        self._lm_result = minimizer.minimize(method='least_squares',
                                             verbose=verbose,
                                             max_nfev=max_nfev)

        self._finalize()

    @property
    def nnls(self) -> bool:
        """bool: If `True` non-linear least squaes optimizing is used instead of variable
        projection."""
        return self._nnls

    @property
    def data(self) -> typing.Dict[str, xr.Dataset]:
        """dict(str, xarray.Dataset): The resulting data as a dictionary of `xarray.Dataset`.

        Note
        ----
        The actual content of the data depends on the actual model and can be found in the
        documentation for the model.
        """
        return self._data

    @property
    def nfev(self) -> int:
        """int: The number of function evaluations."""
        return self._lm_result.nfev if self._lm_result else 0

    @property
    def nvars(self) -> int:
        """int: Number of variables in optimization."""
        return self._lm_result.nvarys if self._lm_result else None

    @property
    def ndata(self) -> int:
        """int: Number of data points."""
        return self._lm_result.ndata if self._lm_result else None

    @property
    def nfree(self) -> int:
        """int: Degrees of freedom in optimization. """
        return self._lm_result.nfree if self._lm_result else None

    @property
    def chisqr(self) -> float:
        """float: The chi-square of the optimization """
        return self._lm_result.chisqr if self._lm_result else 0

    @property
    def red_chisqr(self) -> float:
        """float: The reduced chi-square of the optimization."""
        return self._lm_result.redchi if self._lm_result else 0

    @property
    def root_mean_sqare_error(self) -> float:
        """float: The root mean square error the optimization."""
        return self._lm_result.redchi if self._lm_result else 0

    @property
    def var_names(self) -> typing.List[str]:
        """list(str): Ordered list of variable parameter names used in optimization, and
        useful for understanding the values in :attr:`covar`."""
        return [n.replace('_', '.') for n in self._lm_result.var_names] \
            if self._lm_result else None

    @property
    def covar(self) -> np.ndarray:
        """np.ndarray: Covariance matrix from minimization, with rows and columns
        corresponding to :attr:`var_names`."""
        return self._lm_result.covar if self._lm_result else None

    @property
    def best_fit_parameter(self) -> ParameterGroup:
        """glotaran.model.ParameterGroup: The best fit parameters."""
        if self._lm_result is None:
            return self.initial_parameter
        return ParameterGroup.from_parameter_dict(self._lm_result.params)

    @property
    def initial_parameter(self) -> ParameterGroup:
        """glotaran.model.ParameterGroup: The initital fit parameter"""
        return self._initial_parameter

    @property
    def global_clp(self) -> typing.Dict[any, xr.DataArray]:
        """A dictonary of the global condionally linear parameter with the index on the global
        estimated axis as keys."""
        return self._global_clp

    def _get_group_indices(self, dataset_label):
        return [index for index, item in self._group.items()
                if dataset_label in [val[1].label for val in item]]

    def _get_dataset_idx(self, index, dataset):
            datasets = [val[1].label for val in self._group[index]]
            return datasets.index(dataset)

    def _iter_cb(self, params, i, resid, *args, **kws):
        pass

    def _calculate_residual(self, parameter):

        if not isinstance(parameter, ParameterGroup):
            parameter = ParameterGroup.from_parameter_dict(parameter)

        penalty = []
        for index, item in self._group.items():
            clp_labels, matrix = calculate_group_item(item, self.model, parameter, self._data)

            clp = None
            residual = None
            if self.nnls:
                clp, residual = residual_nnls(
                        matrix,
                        self._data_group[index]
                    )
            else:
                clp, residual = residual_variable_projection(
                        matrix,
                        self._data_group[index]
                    )

            self._global_clp[index] = xr.DataArray(clp, coords=[('clp_label', clp_labels)])

            start = 0
            for i, dataset in item:
                dataset = self._data[dataset.label]
                if 'residual' not in dataset:
                    dataset['residual'] = dataset.data.copy()
                end = dataset.coords[self.model.calculated_axis].size + start
                dataset.residual.loc[{self.model.estimated_axis: i}] = residual[start:end]
                start = end

                if 'clp' not in dataset:
                    dim1 = dataset.coords[self.model.estimated_axis].size
                    dim2 = dataset.coords['clp_label'].size
                    dataset['clp'] = (
                        (self.model.estimated_axis, 'clp_label'),
                        np.zeros((dim1, dim2), dtype=np.float64)
                    )
                dataset.clp.loc[{self.model.estimated_axis: i}] = \
                    np.array([clp[clp_labels.index(i)] if i in clp_labels else None
                              for i in dataset.coords['clp_label'].values])

            if self.model.additional_penalty_function:
                additionals = self.model.additional_penalty_function(
                    parameter, clp_labels, clp, matrix, parameter)
                residual = np.concatenate((residual, additionals))

            penalty.append(residual)

        return np.concatenate(penalty)

    def _finalize(self):
        for label in self.model.dataset:
            dataset = self._data[label]

            if 'weight' in dataset:
                dataset['weighted_residual'] = dataset.residual
                dataset.residual = np.multiply(dataset.weighted_residual, dataset.weight**-1)

            l, v, r = np.linalg.svd(dataset.residual)

            dataset['residual_left_singular_vectors'] = \
                ((self.model.calculated_axis, 'left_singular_value_index'), l)

            dataset['residual_right_singular_vectors'] = \
                ((self.model.estimated_axis, 'right_singular_value_index'), r)

            dataset['residual_singular_values'] = \
                ((self.model.estimated_axis, 'singular_value_index'), r)

            # reconstruct fitted data

            dataset['fitted_data'] = dataset.data - dataset.residual

        if callable(self.model.finalize_result):
            self.model.finalize_result(self)

    def __str__(self):
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
        string += f"{self.chisqr:.6f} |".rjust(lr)
        string += "\n"
        string += "Reduced Chi Square |".rjust(ll)
        string += f"{self.red_chisqr:.9f} |".rjust(lr)
        string += "\n"

        string += "\n"
        string += "## Best Fit Parameter\n\n"
        string += f"{self.best_fit_parameter}"
        string += "\n"

        return string
