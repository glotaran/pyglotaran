import collections
import inspect
import numpy as np
import dask as ds
import dask.array as da
import xarray as xr
import lmfit

from glotaran.parameter import ParameterGroup

from .scheme import Scheme
from .result import Result
from .nnls import residual_nnls
from .variable_projection import residual_variable_projection


Problem = collections.namedtuple('Problem', 'index dataset_descriptor axis')


def _fill_problem(problem, model, parameter):
    if isinstance(problem, list):
        filled = []
        for p in problem:
            filled.append(Problem(
                p.index,
                p.dataset_descriptor.fill(model, parameter),
                p.axis
            ))
        return filled
    return Problem(
        problem.index,
        problem.dataset_descriptor.fill(model, parameter),
        problem.axis
    )


def _calculate_matrix(matrix_function, dataset_descriptor, axis, extra, index=None):
    args = {
        'dataset_descriptor': dataset_descriptor,
        'axis': axis,
    }
    for k, v in extra:
        args[k] = v
    if index is not None:
        args['index'] = index
    return ds.delayed(matrix_function, nout=2)(**args)


def _calculate_problem(matrix_function, problem, extra):
    if isinstance(problem, list):
        clp_labels = []
        matrices = []
        for p in problem:
            clp_label, matrix = _calculate_matrix(
                matrix_function, p.dataset_descriptor, p.axis, extra, index=p.index)
            clp_labels.append(clp_label)
            matrices.append(matrix)
        return clp_labels, matrices
    return _calculate_matrix(
        matrix_function, problem.dataset_descriptor, problem.axis, extra, index=problem.index)


@ds.delayed(nout=2)
def _apply_constraints(constrain_function, parameter, index, clp, matrix):
    return constrain_function(parameter, clp, matrix, index)


@ds.delayed(nout=2)
def _combine_matrices(all_clp, matrices):
    masks = []
    full_clp = None
    for clp in all_clp:
        if full_clp is None:
            full_clp = clp
            masks.append([i for i, _ in enumerate(clp)])
        else:
            mask = []
            for c in clp:
                if c not in full_clp:
                    full_clp.append(c)
                mask.append(full_clp.index(c))
            masks.append(mask)
    dim1 = np.sum([m.shape[0] for m in matrices])
    dim2 = len(full_clp)
    matrix = np.zeros((dim1, dim2), dtype=np.float64)
    start = 0
    for i, m in enumerate(matrices):
        end = start + m.shape[0]
        matrix[start:end, masks[i]] = m
        start = end

    return full_clp, matrix


@ds.delayed(nout=2)
def _calculate_residual(residual_function, matrix, data):
    return residual_function(matrix, data)


@ds.delayed
def _calculate_additional_penalty(penalty_function, parameter, index,
                                  clp_label, clp, matrix, residual):
    return penalty_function(parameter, clp_label, clp, matrix, index)


@ds.delayed
def _concat(penalty):
    return np.concatenate(penalty)


class Optimizer:
    def __init__(self, scheme: Scheme):
        self._scheme = scheme
        self.matrices = {}
        self.clp = {}
        self.clp_label = {}
        self.full_clp = {}
        self.full_clp_label = {}
        self.full_matrices = {}
        self.residual = {}
        self._global_data = {}
        self._global_problem = {}
        self._matrix_axis = {}
        self._matrix_extra = {}
        self._analyze_matrix_function()
        self._create_global_problem()
        self._create_global_data()

    def optimize(self, verbose=True):
        parameter = self._scheme.parameter.as_parameter_dict()
        minimizer = lmfit.Minimizer(
            self._calculate_penalty,
            parameter,
            fcn_args=None,
            fcn_kws=None,
            iter_cb=None,
            scale_covar=True,
            nan_policy='omit',
            reduce_fcn=None,
            **{})
        verbose = 2 if verbose else 0
        lm_result = minimizer.minimize(
            method='least_squares', verbose=verbose, max_nfev=self._scheme.nfev)

        self._optimal_parameter = ParameterGroup.from_parameter_dict(lm_result.params)
        self._calculate_result()

        covar = lm_result.covar if hasattr(lm_result, 'covar') else None

        return Result(self._scheme, self._result_data, self._optimal_parameter,
                      lm_result.nfev, lm_result.nvarys, lm_result.ndata, lm_result.nfree,
                      lm_result.chisqr, lm_result.redchi, lm_result.var_names, covar)

    def _analyze_matrix_function(self):
        self._matrix_function = self._scheme.model.matrix
        signature = inspect.signature(self._matrix_function).parameters
        self._index_dependend = 'index' in signature

    def _create_global_problem(self):

        self._grouped = self._scheme.model.allow_grouping and len(self._scheme.data) != 1

        for label, dataset_descriptor in self._scheme.model.dataset.items():
            if label not in self._scheme.data:
                raise Exception(f"Missing data for dataset '{label}'")

            self._matrix_axis[label] = ds.delayed(
                self._scheme.data[label].coords[self._scheme.model.matrix_dimension].values
            )
            axis = self._scheme.data[label].coords[self._scheme.model.global_dimension].values

            for index in axis:
                if self._grouped:
                    # check if already have an item in the group
                    problem_index = np.where(np.isclose(list(self._global_problem.keys()), index,
                                             atol=self._scheme.group_tolerance))[0]
                    if len(problem_index) == 0:
                        # new index
                        self._global_problem[index] = \
                                [Problem(index, dataset_descriptor, self._matrix_axis[label])]
                    else:
                        # known index
                        idx = list(self._global_problem.keys())[problem_index[0]]
                        self._global_problem[idx]\
                            .append(Problem(index, dataset_descriptor, self._matrix_axis[label]))
                else:
                    self._global_problem[f"{label}_{index}"] =\
                        Problem(index, dataset_descriptor, self._matrix_axis[label])

    def _create_global_data(self):
        self._result_data = self._scheme.prepared_data()
        for index, problem in self._global_problem.items():
            if isinstance(problem, list):
                data = [da.from_array(self._result_data[p.dataset_descriptor.label]
                        .data.sel({self._scheme.model.global_dimension: p.index}).values,
                                      chunks='auto')
                        for p in problem]
                self._global_data[index] = da.concatenate(data).persist()
            else:
                data = self._result_data[problem.dataset_descriptor.label].data
                data = data.sel({self._scheme.model.global_dimension: problem.index}).values
                self._global_data[index] = ds.delayed(data).persist()
                #  self._global_data[index] = da.from_array(data, chunks='auto').persist()

    def _create_calculate_penalty_job(self, parameter: ParameterGroup):

        penalty = []
        matrix = None

        if not self._index_dependend:
            for label, dataset_descriptor in self._scheme.model.dataset.items():
                dataset_descriptor = dataset_descriptor.fill(self._scheme.model, parameter)
                self.clp_label[label], self.matrices[label] = _calculate_matrix(
                    self._matrix_function, dataset_descriptor,
                    self._matrix_axis[label], self._matrix_extra
                )

        for index, problem in self._global_problem.items():

            problem = _fill_problem(problem, self._scheme.model, parameter)

            if self._index_dependend:
                clp_label, matrix = _calculate_problem(
                    self._matrix_function, problem, self._matrix_extra)
                self.clp_label[index] = clp_label
                self.matrices[index] = matrix
                if self._grouped:
                    clp_label, matrix = _combine_matrices(clp_label, matrix)
            else:
                if self._grouped:
                    clp_label, matrix = _combine_matrices(
                        [self.clp_label[p.dataset_descriptor.label] for p in problem],
                        [self.matrices[p.dataset_descriptor.label] for p in problem])
                else:
                    clp_label = self.clp_label[problem.dataset_descriptor.label]
                    matrix = self.matrices[problem.dataset_descriptor.label]

            if callable(self._scheme.model.constrain_matrix_function):
                clp_label, matrix = _apply_constraints(
                    self._scheme.model.constrain_matrix_function,
                    parameter, index, clp_label, matrix
                )

            self.full_clp_label[index] = clp_label
            self.full_matrices[index] = matrix

            residual_function = residual_nnls\
                if self._scheme.nnls else residual_variable_projection
            clp, residual = _calculate_residual(
                residual_function, matrix, self._global_data[index])
            self.full_clp[index] = clp
            self.residual[index] = residual
            penalty.append(residual)

            if callable(self._scheme.model.additional_penalty_function):
                penalty.append(_calculate_additional_penalty(
                    self._scheme.model.additional_penalty_function,
                    parameter, index, clp_label, clp, matrix, residual
                ))

        return _concat(penalty)

    def _calculate_penalty(self, parameter):

        if not isinstance(parameter, ParameterGroup):
            parameter = ParameterGroup.from_parameter_dict(parameter)

        job = self._create_calculate_penalty_job(parameter)

        return job.compute()

    def _calculate_result(self):
        clp_labels, matrices, full_clp_label, full_clp, residuals = ds.compute(
            self.clp_label, self.matrices, self.full_clp_label, self.full_clp, self.residual
        )
        for label, dataset in self._result_data.items():
            if self._index_dependend:
                clp_label = None
                matrix = []
                for index, problem in self._global_problem.items():
                    if isinstance(problem, list):
                        for i, p in enumerate(problem):
                            if p.dataset_descriptor.label == label:
                                matrix.append(matrices[index][i])
                                if clp_label is None:
                                    clp_label = clp_labels[index][i]
                    else:
                        if problem.dataset_descriptor.label == label:
                            matrix.append(matrices[index])
                            if clp_label is None:
                                clp_label = clp_labels[index]
                dataset.coords['clp_label'] = clp_label
                dataset['matrix'] = ((
                    (self._scheme.model.global_dimension),
                    (self._scheme.model.matrix_dimension),
                    ('clp_label')
                ), matrix)
            else:
                clp_label = clp_labels[label]
                matrix = matrices[label]
                dataset.coords['clp_label'] = clp_label
                dataset['matrix'] = ((
                    (self._scheme.model.matrix_dimension),
                    ('clp_label')
                ), matrix)

            residual = []
            dim1 = dataset.coords[self._scheme.model.global_dimension].size
            dim2 = dataset.coords['clp_label'].size
            dataset['clp'] = (
                (self._scheme.model.global_dimension, 'clp_label'),
                np.zeros((dim1, dim2), dtype=np.float64))
            for index, problem in self._global_problem.items():
                if isinstance(problem, list):
                    for i, p in enumerate(problem):
                        start = 0
                        if p.dataset_descriptor.label == label:
                            end = start + dataset.coords[self._scheme.model.matrix_dimension].size
                            dataset.clp.loc[{self._scheme.model.global_dimension: p.index}] = \
                                np.array([full_clp[index][full_clp_label[index].index(i)]
                                          if i in full_clp_label[index] else None
                                          for i in dataset.coords['clp_label'].values])
                            residual.append(residuals[index][start:end])
                        else:
                            start += self._result_data[p.dataset_descriptor.label]\
                                    .coords[self._scheme.model.matrix_dimension].size
                else:
                    if problem.dataset_descriptor.label == label:
                        dataset.clp.loc[{self._scheme.model.global_dimension: index}] = \
                            np.array([full_clp[index][full_clp_label[index].index(i)]
                                      if i in full_clp_label[index] else None
                                      for i in dataset.coords['clp_label'].values])
                        residual.append(residuals[index])
            dataset['residual'] = ((
                (self._scheme.model.matrix_dimension),
                (self._scheme.model.global_dimension),
            ), np.asarray(residual).T)

            if 'weight' in dataset:
                dataset['weighted_residual'] = dataset.residual
                dataset.residual = np.multiply(dataset.weighted_residual, dataset.weight**-1)

            l, v, r = np.linalg.svd(dataset.residual)

            dataset['residual_left_singular_vectors'] = \
                ((self._scheme.model.matrix_dimension, 'left_singular_value_index'), l)

            dataset['residual_right_singular_vectors'] = \
                (('right_singular_value_index', self._scheme.model.global_dimension), r)

            dataset['residual_singular_values'] = \
                (('singular_value_index'), v)

            # reconstruct fitted data

            dataset['fitted_data'] = dataset.data - dataset.residual

        if callable(self._scheme.model.finalize_data):
            global_clp = {index: xr.DataArray(clp, coords=[('clp_label', full_clp_label[index])])
                          for index, clp in full_clp.items()}
            self._scheme.model.finalize_data(
                global_clp, self._optimal_parameter, self._result_data)
