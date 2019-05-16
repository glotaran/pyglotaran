import numpy as np

from glotaran.model import Model, model, model_attribute


def calculate_c(dataset_descriptor=None, axis=None, index=None):
    compartments = ['s1', 's2']
    r_compartments = []
    array = np.zeros((axis.shape[0], len(compartments)))

    for i in range(len(compartments)):
        r_compartments.append(compartments[i])
        for j in range(axis.shape[0]):
            array[j, i] = (i + j) * axis[j]
    return (r_compartments, array)


def calculate_e(dataset, axis):
    compartments = ['s1', 's2']
    r_compartments = []
    array = np.zeros((axis.shape[0], len(compartments)))

    for i in range(len(compartments)):
        r_compartments.append(compartments[i])
        for j in range(axis.shape[0]):
            array[j, i] = (i + j) * axis[j]
    return (r_compartments, array)


@model_attribute(
    properties={
        'grouped': bool,
        'indexdependend': bool,
    }
)
class MockMegacomplex:
    pass


@model('mock',
       matrix=calculate_c,
       matrix_dimension='c',
       global_matrix=calculate_e,
       global_dimension='e',
       megacomplex_type=MockMegacomplex,
       )
class MockModel(Model):
    pass
