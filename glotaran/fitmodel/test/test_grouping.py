import pytest
import numpy as np

from glotaran.model import Dataset, Model, ParameterGroup, glotaran_model
from glotaran.fitmodel.grouping import create_group, calculate_group, get_data_group


def calculate(dataset, index, axis):
    compartments = []
    array = np.zeros((len(dataset.initial_concentration.parameters), axis.shape[0]))

    for i in range(len(dataset.initial_concentration.parameters)):
        compartments.append(f's{i}')
        for j in range(axis.shape[0]):
            array[i, j] = dataset.initial_concentration.parameters[i] * axis[j]
    return (compartments, array)


class MockDataset(Dataset):

    def __init__(self, est_axis, calc_axis):
        super(MockDataset, self).__init__()
        self.set_axis('e', est_axis)
        self.set_axis('c', calc_axis)
        self.set_estimated_axis('e')
        self.set_calculated_axis('c')
        self.set(np.ones((len(est_axis), len(calc_axis))))


@glotaran_model('mock',
                calculated_matrix=calculate,
                estimated_matrix=calculate,
                )
class MockModel(Model):
    pass


def test_single_dataset():
    model = MockModel.from_dict({
        "initial_concentration": {
            "j1": [["1", "2"]]
        },
        "dataset": {
            "dataset1": {
                "initial_concentration": 'j1',
                "megacomplex": [],
            },
        }
    })
    print(model.errors())
    assert model.valid()

    parameter = ParameterGroup.from_list([1, 10])
    print(model.errors_parameter(parameter))
    assert model.valid_parameter(parameter)

    model.set_data('dataset1', MockDataset([1, 2, 3], [5, 7, 9, 12]))

    group = create_group(model)
    assert len(group) == 3
    assert [item[0][0] for _, item in group.items()] == [1, 2, 3]
    assert all([item[0][1].label == 'dataset1' for _, item in group.items()])

    result = calculate_group(group, model, parameter)
    assert len(result) == 3
    print(result[0])
    assert result[0].shape == (2, 4)

    data = get_data_group(group)
    assert len(data) == 3
    assert data[0].shape[0] == 4 

    group = create_group(model, group_axis='calculated')
    assert len(group) == 4
    assert [item[0][0] for _, item in group.items()] == [5, 7, 9, 12]
    assert all([item[0][1].label == 'dataset1' for _, item in group.items()])

    result = calculate_group(group, model, parameter, matrix='estimated')
    assert len(result) == 4
    print(result[0])
    assert result[0].shape == (2, 3)


def test_multi_dataset_no_overlap():
    model = MockModel.from_dict({
        "initial_concentration": {
            "j1": [["1", "2"]]
        },
        "dataset": {
            "dataset1": {
                "initial_concentration": 'j1',
                "megacomplex": [],
            },
            "dataset2": {
                "initial_concentration": 'j1',
                "megacomplex": [],
            },
        }
    })

    print(model.errors())
    assert model.valid()

    parameter = ParameterGroup.from_list([1, 10])
    print(model.errors_parameter(parameter))
    assert model.valid_parameter(parameter)

    model.set_data('dataset1', MockDataset([1, 2, 3], [5, 7]))
    model.set_data('dataset2', MockDataset([4, 5, 6], [5, 7, 9]))

    group = create_group(model)
    assert len(group) == 6
    assert [item[0][0] for _, item in group.items()] == [1, 2, 3, 4, 5, 6]
    assert [item[0][1].label for _, item in group.items()] == \
        ['dataset1' for _ in range(3)] + ['dataset2' for _ in range(3)]

    result = calculate_group(group, model, parameter)
    assert len(result) == 6
    print(result[0])
    assert result[0].shape == (2, 2)
    assert result[3].shape == (2, 3)

    data = get_data_group(group)
    assert len(data) == 6
    assert data[0].shape[0] == 2 
    assert data[3].shape[0] == 3 


def test_multi_dataset_overlap():
    model = MockModel.from_dict({
        "initial_concentration": {
            "j1": [["1", "2"]]
        },
        "dataset": {
            "dataset1": {
                "initial_concentration": 'j1',
                "megacomplex": [],
            },
            "dataset2": {
                "initial_concentration": 'j1',
                "megacomplex": [],
            },
        }
    })

    print(model.errors())
    assert model.valid()

    parameter = ParameterGroup.from_list([1, 10])
    print(model.errors_parameter(parameter))
    assert model.valid_parameter(parameter)

    model.set_data('dataset1', MockDataset([0, 1, 2, 3], [5, 7]))
    model.set_data('dataset2', MockDataset([1.4, 2.4, 3.4, 9], [5, 7, 9, 12]))

    group = create_group(model)
    assert len(group) == 5
    assert group[0][0][1].label == 'dataset1'
    assert group[1][0][1].label == 'dataset1'
    assert group[1][1][1].label == 'dataset2'
    assert group[9][0][1].label == 'dataset2'

    result = calculate_group(group, model, parameter)
    assert len(result) == 5
    print(result[0])
    assert result[0].shape == (2, 2)
    assert result[1].shape == (2, 6)
    assert result[4].shape == (2, 4)

    data = get_data_group(group)
    assert len(data) == 5
    assert data[0].shape[0] == 2 
    assert data[1].shape[0] == 6 
    assert data[4].shape[0] == 4

def test_calculate():
    model = MockModel.from_dict({
        "dataset": {
            "dataset1": {
                "initial_concentration": 'j1',
                "megacomplex": ['m1', 'm2'],
            },
            "dataset2": {
                "initial_concentration": 'j1',
                "megacomplex": ['m1', 'm2'],
            },
        }
    })

    model.set_data('dataset1', MockDataset([1, 2, 3], [5, 7]))
    model.set_data('dataset2', MockDataset([1.4, 2.4, 3.4], [5, 7]))

    group = create_group(model)
    assert len(group) == 3
    assert [item[0][0] for _, item in group.items()] == [1, 2, 3]
    assert all([item[0][1].label == 'dataset1' for _, item in group.items()])
    assert all([item[1][1].label == 'dataset2' for _, item in group.items()])
