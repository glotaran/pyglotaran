import lmfit
import numpy as np

from glotaran.parameter import ParameterGroup

from .problem import Problem
from .result import Result
from .scheme import Scheme


def optimize(scheme: Scheme, verbose: bool = True) -> Result:

    initial_parameter = scheme.parameter.as_parameter_dict()
    problem = Problem(scheme)

    minimizer = lmfit.Minimizer(
        _calculate_penalty,
        initial_parameter,
        fcn_args=[problem],
        fcn_kws=None,
        iter_cb=None,
        scale_covar=True,
        nan_policy="omit",
        reduce_fcn=None,
        **{},
    )
    verbose = 2 if verbose else 0
    lm_result = minimizer.minimize(method="least_squares", verbose=verbose, max_nfev=scheme.nfev)

    _create_result(problem)
    covar = lm_result.covar if hasattr(lm_result, "covar") else None

    return Result(
        scheme,
        problem.datasets,
        problem.parameter,
        lm_result.nfev,
        lm_result.nvarys,
        lm_result.ndata,
        lm_result.nfree,
        lm_result.chisqr,
        lm_result.redchi,
        lm_result.var_names,
        covar,
    )


def _calculate_penalty(parameter: lmfit.Parameters, problem: Problem):
    problem.parameter = ParameterGroup.from_parameter_dict(parameter)
    return problem.full_penalty


def _create_result(problem: Problem):

    for label, dataset in problem.data.items():
        if problem.index_dependent:
            if problem.grouped:
                for i, grouped_problem in enumerate(problem.bag):
                    if label in grouped_problem:
                        group_index = [
                            descriptor.label for descriptor in grouped_problem.descriptor
                        ].index(label)
                        if "clp_label" not in dataset.coords:
                            # we assume that the labels are the same, this might not be true in
                            # future models
                            dataset.coords["clp_label"] = problem.clp_labels[i][group_index]

                        if "matrix" not in dataset:
                            dim1 = dataset.coords[problem.model.global_dimension].size
                            dim2 = dataset.coords[problem.model.model_dimension].size
                            dim3 = dataset.clp_label.size
                            dataset["matrix"] = (
                                (
                                    (problem.model.global_dimension),
                                    (problem.model.model_dimension),
                                    ("clp_label"),
                                ),
                                np.zeros((dim1, dim2, dim3), dtype=np.float64),
                            )

                        if "clp" not in dataset:
                            dim1 = dataset.coords[problem.model.global_dimension].size
                            dim2 = dataset.clp_label.size
                            dataset["clp"] = (
                                (
                                    (problem.model.global_dimension),
                                    ("clp_label"),
                                ),
                                np.zeros((dim1, dim2, dim3), dtype=np.float64),
                            )

                        if "residual" not in dataset:
                            dim1 = dataset.coords[problem.model.model_dimension].size
                            dim2 = dataset.coords[problem.model.global_dimension].size
                            dataset["weighted_residual"] = (
                                (problem.model.model_dimension, problem.model.global_dimension),
                                np.zeros((dim1, dim2), dtype=np.float64),
                            )
                            dataset["residual"] = (
                                (problem.model.model_dimension, problem.model.global_dimension),
                                np.zeros((dim1, dim2), dtype=np.float64),
                            )

                        index = grouped_problem.descriptor[group_index].index
                        dataset.matrix.loc[
                            {problem.model.global_dimension: index}
                        ] = problem.matrices[i][group_index]

                        for j, clp in problem.full_clps:
                            dataset.clp.loc[
                                {
                                    problem.model.global_dimension: index,
                                    "clp_label": problem.clp_labels[i][j],
                                }
                            ] = clp
                    start = 0
                    for j in range(group_index):
                        start += (
                            problem.data[grouped_problem.descriptor[j].label]
                            .coords[problem.model.model_dimension]
                            .size
                        )
                    end = start + dataset.coords[problem.model.model_dimension].size
                    dataset.weighted_residual.loc[
                        {problem.model.global_dimension: index}
                    ] = problem.weighted_residuals[i][start:end]
                    dataset.residual.loc[
                        {problem.model.global_dimension: index}
                    ] = problem.residuals[i][start:end]
            else:
                dataset.coords["clp_label"] = problem.clp_labels[label][0]
                dataset["matrix"] = (
                    (
                        (problem.model.global_dimension),
                        (problem.model.model_dimension),
                        ("clp_label"),
                    ),
                    np.asarray(problem.matrices[label]),
                )
                dataset["clp"] = (
                    (
                        (problem.model.global_dimension),
                        ("clp_label"),
                    ),
                    np.asarray(problem.full_clps[label]),
                )
                dataset["weighted_residual"] = (
                    (
                        (problem.model.model_dimension),
                        (problem.model.global_dimension),
                    ),
                    np.asarray(problem.weighted_residuals[label]),
                )
                dataset["residual"] = (
                    (
                        (problem.model.model_dimension),
                        (problem.model.global_dimension),
                    ),
                    np.asarray(problem.residuals[label]),
                )
        else:
            dataset.coords["clp_label"] = problem.clp_labels[label]
            dataset["matrix"] = (
                (
                    (problem.model.model_dimension),
                    ("clp_label"),
                ),
                np.asarray(problem.matrices[label]),
            )
            dataset["clp"] = (
                (
                    (problem.model.global_dimension),
                    ("clp_label"),
                ),
                np.asarray(problem.full_clps[label]),
            )
            dataset["weighted_residual"] = (
                (
                    (problem.model.model_dimension),
                    (problem.model.global_dimension),
                ),
                np.asarray(problem.weighted_residuals[label]),
            )
            dataset["residual"] = (
                (
                    (problem.model.model_dimension),
                    (problem.model.global_dimension),
                ),
                np.asarray(problem.residuals[label]),
            )
        _create_svd("weighted_residual", dataset, problem.model)
        _create_svd("residual", dataset, problem.model)

        # Calculate RMS
        size = dataset.residual.shape[0] * dataset.residual.shape[1]
        dataset.attrs["root_mean_square_error"] = np.sqrt(
            (dataset.residual ** 2).sum() / size
        ).values

        # reconstruct fitted data
        dataset["fitted_data"] = dataset.data - dataset.residual

    if callable(problem.model.finalize_data):
        problem.model.finalize_data(problem)


def _create_svd(name, dataset, model):
    l, v, r = np.linalg.svd(dataset[name])

    dataset[f"{name}_left_singular_vectors"] = (
        (model.model_dimension, "left_singular_value_index"),
        l,
    )

    dataset[f"{name}_right_singular_vectors"] = (
        ("right_singular_value_index", model.global_dimension),
        r,
    )

    dataset[f"{name}_singular_values"] = (("singular_value_index"), v)
