import numpy as np


def retrieve_sas(result, dataset):
    labels, clp = result.get_clp(dataset)
    dataset = result.model.dataset[dataset].fill(result.model, result.best_fit_parameter)
    result = {}
    for cmplx, matrix in dataset.get_k_matrices():
        compartments = matrix.involved_compartments()
        compartments = [c for c in dataset.initial_concentration.compartments if c in compartments]
        dim1 = clp.shape[0]
        dim2 = len(compartments)
        sas = np.zeros((dim1, dim2), dtype=np.float64)
        for i, comp in enumerate(compartments):
            sas[:, i] = clp[:, labels.index(comp)]
        result[cmplx] = (compartments, sas)
    return result


def retrieve_das(result, dataset):
    labels, clp = result.get_clp(dataset)
    dataset = result.model.dataset[dataset].fill(result.model, result.best_fit_parameter)
    result = {}
    for cmplx, matrix in dataset.get_k_matrices():
        compartments = matrix.involved_compartments()
        compartments = [c for c in dataset.initial_concentration.compartments if c in compartments]
        dim1 = clp.shape[0]
        dim2 = len(compartments)
        sas = np.zeros((dim1, dim2), dtype=np.float64)
        for i, comp in enumerate(compartments):
            sas[:, i] = clp[:, labels.index(comp)]
        a_matrix = matrix.a_matrix(dataset.initial_concentration)
        das = np.dot(sas, a_matrix)
        result[cmplx] = (compartments, das)

    return result
