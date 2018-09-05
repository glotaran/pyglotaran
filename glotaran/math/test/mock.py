import numpy as np

from glotaran.model import Dataset, Model, ParameterGroup, glotaran_model


def calculate_c(dataset, compartments, index, axis):
    r_compartments = []
    array = np.zeros((len(dataset.initial_concentration.parameters), axis.shape[0]))

    for i in range(len(dataset.initial_concentration.parameters)):
        r_compartments.append(compartments[i])
        for j in range(axis.shape[0]):
            array[i, j] = dataset.initial_concentration.parameters[i] * axis[j]
    return (r_compartments, array)


def calculate_e(dataset, compartments, axis):
    return calculate_c(dataset, compartments, 0, axis)[1].T


class MockDataset(Dataset):

    def __init__(self, est_axis, calc_axis):
        super(MockDataset, self).__init__()
        self.set_axis('e', est_axis)
        self.set_axis('c', calc_axis)
        self.set(np.ones((len(est_axis), len(calc_axis))))


@glotaran_model('mock',
                calculated_matrix=calculate_c,
                calculated_axis='c',
                estimated_matrix=calculate_e,
                estimated_axis='e',
                )
class MockModel(Model):
    pass
