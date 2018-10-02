import numpy as np

from ...model import Dataset, BaseModel, model
# from ...model.model import model


def calculate_c(dataset, compartments, index, axis):
    r_compartments = []
    array = np.zeros((len(compartments), axis.shape[0]))

    for i in range(len(compartments)):
        r_compartments.append(compartments[i])
        for j in range(axis.shape[0]):
            array[i, j] = (i + j) * axis[j]
    return (r_compartments, array)


def calculate_e(dataset, compartments, axis):
    return calculate_c(dataset, compartments, 0, axis)[1].T


class MockDataset(Dataset):

    def __init__(self, est_axis, calc_axis):
        super(MockDataset, self).__init__()
        self.set_axis('e', np.asarray(est_axis))
        self.set_axis('c', np.asarray(calc_axis))
        self.set_data(np.ones((len(est_axis), len(calc_axis))))


@model('mock',
       calculated_matrix=calculate_c,
       calculated_axis='c',
       estimated_matrix=calculate_e,
       estimated_axis='e',
       )
class MockModel(BaseModel):
    pass
