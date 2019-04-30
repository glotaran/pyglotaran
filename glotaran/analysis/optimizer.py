import collections
import itertools
import inspect
import numpy as np
import dask as ds
import dask.array as da
import dask.bag as db
import dask.distributed as dd
import xarray as xr
import lmfit

from glotaran.parameter import ParameterGroup

from .scheme import Scheme
from .result import Result
from .nnls import residual_nnls
from .variable_projection import residual_variable_projection


class ParameterServer:

    def __init__(self):
        self.parameter = None

    def update(self, parameter):
        self.parameter = parameter

    def get(self):
        return self.parameter



LabelAndMatrix = collections.namedtuple('LabelAndMatrix', 'clp_label matrix')
ClpAndResidual = collections.namedtuple('ClpAndResidual', 'clp residual')


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


def _findOverlap(a, b, rtol=1e-05, atol=1e-08):
    ovr_a = []
    ovr_b = []
    start_b = 0
    for i, ai in enumerate(a):
        for j, bj in itertools.islice(enumerate(b), start_b, None):
            if np.isclose(ai, bj, rtol=rtol, atol=atol, equal_nan=False):
                ovr_a.append(i)
                ovr_b.append(j)
            elif bj > ai:  # (more than tolerance)
                break  # all the rest will be farther away
            else:  # bj < ai (more than tolerance)
                start_b += 1  # ignore further tests of this item
    return (ovr_a, ovr_b)


def _calculate_matrix(matrix_function, dataset_descriptor, axis, extra, index=None):
    args = {
        'dataset_descriptor': dataset_descriptor,
        'axis': axis,
    }
    for k, v in extra:
        args[k] = v
    if index is not None:
        args['index'] = index
    clp_label, matrix = matrix_function(**args)
    if dataset_descriptor.scale is not None:
        matrix *= dataset_descriptor.scale
    return LabelAndMatrix(clp_label, matrix)


def _calculate_problem_(matrix_function, problem, extra):
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


@ds.delayed
def _get_dataset_descriptor(dataset_descriptor, model, parameter_server):
    parameter = ParameterGroup.from_parameter_dict(parameter_server.get().result())
    return dataset_descriptor.fill(model, parameter)


def _calculate_grouped_matrices(problem, matrix_function=None, dataset_descriptors=None, extra={}):
    if len(problem.descriptors) == 1:
        descriptor = problem.descriptors[0]
        dataset_descriptor = dataset_descriptors[descriptor.dataset]
        return _calculate_matrix(
            matrix_function, dataset_descriptor, descriptor.axis, extra, index=descriptor.index)
    clp_labels = []
    matrices = []
    for descriptor in problem.descriptors:
        dataset_descriptor = dataset_descriptors[descriptor.dataset]
        clp, matrix = _calculate_matrix(
            matrix_function, dataset_descriptor, descriptor.axis, extra, index=descriptor.index)
        clp_labels.append(clp)
        matrices.append(matrix)
    return LabelAndMatrix(clp_labels, matrices)


@ds.delayed(nout=2)
def _apply_constraints(constrain_function, parameter, index, clp, matrix):
    return constrain_function(parameter, clp, matrix, index)


def _optimize(penalty_job, initial_parameter, parameter_server, nfev, verbose=True):
    def residual(parameter):
        parameter_server.update(parameter).result()
        return penalty_job.compute()

    minimizer = lmfit.Minimizer(
        residual,
        initial_parameter,
        fcn_args=None,
        fcn_kws=None,
        iter_cb=None,
        scale_covar=True,
        nan_policy='omit',
        reduce_fcn=None,
        **{})
    verbose = 2 if verbose else 0
    lm_result = minimizer.minimize(
        method='least_squares', verbose=verbose, max_nfev=nfev)



def _combine_matrices(label_and_matrices):
    (all_clp, matrices) = label_and_matrices
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

    return LabelAndMatrix(full_clp, matrix)


def _calculate_residual(problem, label_and_matrix, residual_function=None):
    clp, residual = residual_function(label_and_matrix.matrix, problem.data)
    return ClpAndResidual(clp, residual)


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
        self._grouped = self._scheme.model.allow_grouping and len(self._scheme.data) != 1
        self._global_data = {}
        self._global_problem = {}
        self._matrix_axis = {}
        self._matrix_extra = {}
        self._analyze_matrix_function()
        #  self._create_global_problem()
        #  self._create_global_data()

    def optimize(self, verbose=True):
        parameter = self._scheme.parameter.as_parameter_dict()

        client = dd.get_client()
        parameter_server = client.submit(ParameterServer, actor=True).result()
        bag = self._create_bag()
        client.persist(bag)

        lams = self._create_index_dependend_matrix_job(bag, parameter_server)

        penalty_job = self._create_residual_job(bag, lams)

        client.submit(
            _optimize, penalty_job, parameter, self._scheme.nfev, verbose
        ).result()

        #  self._optimal_parameter = ParameterGroup.from_parameter_dict(lm_result.params)
        #  self._calculate_result()
        #
        #  covar = lm_result.covar if hasattr(lm_result, 'covar') else None
        #
        #  return Result(self._scheme, self._result_data, self._optimal_parameter,
        #                lm_result.nfev, lm_result.nvarys, lm_result.ndata, lm_result.nfree,
        #                lm_result.chisqr, lm_result.redchi, lm_result.var_names, covar)

    def _analyze_matrix_function(self):
        self._matrix_function = self._scheme.model.matrix
        signature = inspect.signature(self._matrix_function).parameters
        self._index_dependend = 'index' in signature

    def _create_bag(self):
        return self._create_grouped_bag()

    def _create_grouped_bag(self):
        bag = None
        full_axis = None
        for label in self._scheme.model.dataset:
            dataset = self._scheme.data[label]
            global_axis = dataset.coords[self._scheme.model.global_dimension].values
            model_axis = dataset.coords[self._scheme.model.matrix_dimension].values
            if bag is None:
                bag = collections.deque(
                    Problem(dataset.data.isel({self._scheme.model.global_dimension: i}),
                            [ProblemDescriptor(label, value, model_axis)])
                    for i, value in enumerate(global_axis)
                )
                full_axis = collections.deque(global_axis)
            else:
                i1, i2 = _findOverlap(full_axis, global_axis, atol=0.1)

                for i, j in enumerate(i1):
                    bag[j] = Problem(
                        da.concatenate([bag[j][0], dataset.data.isel(
                            {self._scheme.model.global_dimension: i2[i]})]),
                        bag[j][1] + [ProblemDescriptor(label,
                                                       global_axis[i2[i]], model_axis)]
                    )

                for i in range(0, i2[0]):
                    full_axis.appendleft(global_axis[i2[i]])
                    bag.appendleft(Problem(
                        dataset.data.isel({self._scheme.model.global_dimension: i2[i]}),
                        [ProblemDescriptor(label, global_axis[i], model_axis)]
                    ))

                for i in range(i2[-1]+1, len(global_axis)):
                    full_axis.append(global_axis[i])
                    bag.append(Problem(
                        dataset.data.isel({self._scheme.model.global_dimension: i2[i]}),
                        [ProblemDescriptor(label, global_axis[i], model_axis)]
                    ))

        return db.from_sequence(bag)

    def _create_dataset_descriptor_futures(self, parameter_server):
        futures = {}
        for label, dataset in self._scheme.model.dataset.items():
            futures[label] = _get_dataset_descriptor(dataset, self._scheme.model, parameter_server)
        return futures

    def _create_index_dependend_matrix_job(self, bag, parameter_server):
        datasets = self._create_dataset_descriptor_futures(parameter_server)
        lams = bag.map(_calculate_grouped_matrices,
                       matrix_function=self._scheme.model.matrix,
                       dataset_descriptors=datasets,
                       )

        self._label_and_matrices = lams

        if self._grouped:
            lams = lams.map(_combine_matrices)
        return lams

    def _create_residual_job(self, bag, lams):

        residual_function = residual_nnls\
            if self._scheme.nnls else residual_variable_projection
        self._clp_and_residuals = \
            bag.map(
                _calculate_residual,
                lams,
                residual_function=residual_function)
        return self._clp_and_residuals.reduction(
            lambda cars: np.concatenate([car.residual for car in cars]),
            lambda residual: np.concatenate(list(residual)),
        )

    def _create_global_problem(self):


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
                    start = 0
                    for i, p in enumerate(problem):
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
                        dataset.clp.loc[{self._scheme.model.global_dimension: problem.index}] = \
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

            size = dataset.residual.shape[0] * dataset.residual.shape[1]
            dataset.attrs['root_mean_square_error'] = \
                np.sqrt((dataset.residual**2).sum()/size).values

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
