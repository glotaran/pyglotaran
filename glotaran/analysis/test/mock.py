import numpy as np

from glotaran.model import Model, model, model_attribute


def calculate_c(dataset, index, axis):
    compartments = ['s1', 's2']
    r_compartments = []
    array = np.zeros((axis.shape[0], len(compartments)))

    for i in range(len(compartments)):
        r_compartments.append(compartments[i])
        for j in range(axis.shape[0]):
            array[j, i] = (i + j) * axis[j]
    return (r_compartments, array)


def calculate_e(dataset, axis):
    return calculate_c(dataset, 0, axis)[1].T


@model_attribute()
class MockMegacomplex:
    pass


@model('mock',
       calculated_matrix=calculate_c,
       calculated_axis='c',
       estimated_matrix=calculate_e,
       estimated_axis='e',
       megacomplex_type=MockMegacomplex,
       )
class MockModel(Model):
    pass
