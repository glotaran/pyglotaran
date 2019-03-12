"""The result class for global analysis."""

import typing
import os

import numpy as np
import xarray as xr
import lmfit

import glotaran  # noqa F01
from glotaran.parameter import ParameterGroup


from .grouping import create_group, create_data_group
from .scheme import Scheme
from .optimize import calculate_residual


class Result:

    def __init__(self, scheme: Scheme):
        """The result of a global analysis.

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
            The tolerance for grouping datasets along the global axis.
        """
        self._scheme = scheme
        self._data = scheme.prepared_data()
        self._group = create_group(scheme.model, scheme.data, scheme.group_tolerance)
        self._data_group = create_data_group(scheme.model, self._group, self._data)
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
        """Creates a :class:`Result` from parameters without optimization.

        Parameters
        ----------
        model :
            A subclass of :class:`glotaran.model.Model`
        data : dict(str, union(xr.Dataset, xr.DataArray))
            A dictonary containing all datasets with their labels as keys.
        parameter :
            The parameter,
        nnls :
            If `True` non-linear least squaes optimizing is used instead of variable projection.
        atol :
            The tolerance for grouping datasets along the global axis.
        """
        scheme = Scheme(model=model, parameter=parameter, data=data,
                        nnls=nnls, group_tolerance=atol)
        cls = cls(scheme)
        calculate_residual(parameter, cls)
        cls.finalize()
        return cls

    @property
    def scheme(self) -> Scheme:
        """The scheme for analysis."""
        return self._scheme

    @property
    def model(self) -> typing.Type['glotaran.model.Model']:
        """The model for analysis."""
        return self._scheme.model

    @property
    def nnls(self) -> bool:
        """If `True` non-linear least squaes optimizing is used instead of variable
        projection."""
        return self._scheme.nnls

    @property
    def data(self) -> typing.Dict[str, xr.Dataset]:
        """The resulting data as a dictionary of :xarraydoc:`Dataset`.

        Notes
        -----
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
        """Number of variables in optimization :math:`N_{vars}`"""
        return self._lm_result.nvarys if self._lm_result else None

    @property
    def ndata(self) -> int:
        """Number of data points :math:`N`."""
        return self._lm_result.ndata if self._lm_result else None

    @property
    def nfree(self) -> int:
        """Degrees of freedom in optimization :math:`N - N_{vars}`."""
        return self._lm_result.nfree if self._lm_result else None

    @property
    def chisqr(self) -> float:
        """The chi-square of the optimization
        :math:`\chi^2 = \sum_i^N [{Residual}_i]^2`.""" # noqa w605
        return self._lm_result.chisqr if self._lm_result else 0

    @property
    def red_chisqr(self) -> float:
        """The reduced chi-square of the optimization
        :math:`\chi^2_{red}= {\chi^2} / {(N - N_{vars})}`.
        """ # noqa w605
        return self._lm_result.redchi if self._lm_result else 0

    @property
    def root_mean_sqare_error(self) -> float:
        """
        The root mean square error the optimization
        :math:`rms = \sqrt{\chi^2_{red}}`
        """ # noqa w605
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
    def optimized_parameter(self) -> ParameterGroup:
        """The optimized parameters."""
        if self._lm_result is None:
            return self.initial_parameter
        return ParameterGroup.from_parameter_dict(self._lm_result.params)

    @property
    def initial_parameter(self) -> ParameterGroup:
        """The initital fit parameter"""
        return self._scheme.parameter

    @property
    def global_clp(self) -> typing.Dict[typing.Any, xr.DataArray]:
        """A dictonary of the global condionally linear parameter with the index on the global
        dimension as keys."""
        return self._global_clp

    @property
    def data_groups(self) -> typing.Dict[typing.Any, np.ndarray]:
        """A dictonary of the data groups along the global axis."""
        return self._data_group

    @property
    def groups(self) -> typing.Dict[typing.Any, typing.List[typing.Tuple[typing.Any, str]]]:
        """A dictonary of the dataset_descriptor groups along the global axis."""
        return self._group

    def get_dataset(self, dataset_label: str) -> xr.Dataset:
        """Returns the result dataset for the given dataset label.

        Parameters
        ----------
        dataset_label :
            The label of the dataset.
        """
        try:
            return self.data[dataset_label]
        except KeyError:
            raise Exception(f"Unknown dataset '{dataset_label}'")

    def finalize(self, lm_result: lmfit.minimizer.MinimizerResult = None):
        """Finalizes the result. Calculates the unweighted residual (if applicable), the residual
        svd and calls the model's finalize function.

        Notes
        -----

        This function is intended for internal use and should not be called by users.

        Parameters
        ----------
        lm_result :
            The result of the optimization with `lmfit`.
        """

        if lm_result:
            self._lm_result = lm_result

        for label in self.model.dataset:
            dataset = self._data[label]

            if 'weight' in dataset:
                dataset['weighted_residual'] = dataset.residual
                dataset.residual = np.multiply(dataset.weighted_residual, dataset.weight**-1)

            l, v, r = np.linalg.svd(dataset.residual)

            dataset['residual_left_singular_vectors'] = \
                ((self.model.matrix_dimension, 'left_singular_value_index'), l)

            dataset['residual_right_singular_vectors'] = \
                (('right_singular_value_index', self.model.global_dimension), r)

            dataset['residual_singular_values'] = \
                ((self.model.global_dimension, 'singular_value_index'), r)

            # reconstruct fitted data

            dataset['fitted_data'] = dataset.data - dataset.residual

        if callable(self.model._finalize_result):
            self.model._finalize_result(self)

    def save(self,  path: str) -> typing.List[str]:
        """Saves the result to given folder.

        Returns a list with paths of all saved items.

        The following files are saved:

        * `result.md`: The result with the model formatted as markdown text.
        * `optimized_parameter.csv`: The optimized parameter as csv file.
        * `{dataset_label}.nc`: The result data for each dataset as NetCDF file.

        Parameters
        ----------
        path :
            The path to the folder in which to save the result.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        elif not os.path.isdir(path):
            raise Exception(f"The path '{path}' is not a directory.")

        paths = []

        md_path = os.path.join(path, 'result.md')
        with open(md_path, 'w') as f:
            f.write(self.markdown())
        paths.append(md_path)

        csv_path = os.path.join(path, 'optimized_parameter.csv')
        self.optimized_parameter.to_csv(csv_path)
        paths.append(csv_path)

        for label, data in self.data.items():
            nc_path = os.path.join(path, f"{label}.nc")
            data.to_netcdf(nc_path, engine='netcdf4')
            paths.append(nc_path)

        return paths

    def markdown(self, with_model=True) -> str:
        """Formats the model as a markdown text.

        Parameters
        ----------
        with_model :
            If `True`, the model will be printed together with the initial and optimized parameter.
        """

        ll = 32
        lr = 13

        string = "Optimization Result".ljust(ll-1)
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

        if with_model:
            string += "\n\n" + self.model.markdown(parameter=self.optimized_parameter,
                                                   initial=self.initial_parameter)

        return string

    def __str__(self):
        return self.markdown(with_model=False)
