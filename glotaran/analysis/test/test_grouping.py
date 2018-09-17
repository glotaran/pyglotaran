
from glotaran.analysis.grouping import create_group, calculate_group, create_data_group
from glotaran.model import ParameterGroup

from .mock import MockDataset, MockModel


def test_single_dataset():
    model = MockModel.from_dict({
        "compartment": ["s1", "s2"],
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

    data = {'dataset1': MockDataset([1, 2, 3], [5, 7, 9, 12])}

    group = create_group(model, data)
    assert len(group) == 3
    assert [item[0][0] for _, item in group.items()] == [1, 2, 3]
    assert all([item[0][1].label == 'dataset1' for _, item in group.items()])

    result = list(calculate_group(group, model, parameter, data))
    assert len(result) == 3
    print(result[0])
    assert result[0][1].shape == (2, 4)

    data = create_data_group(model, group, data)
    assert len(data) == 3
    assert data[0].shape[0] == 4


def test_multi_dataset_no_overlap():
    model = MockModel.from_dict({
        "compartment": ["s1", "s2"],
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

    data = {
        'dataset1': MockDataset([1, 2, 3], [5, 7]),
        'dataset2': MockDataset([4, 5, 6], [5, 7, 9]),
    }

    group = create_group(model, data)
    assert len(group) == 6
    assert [item[0][0] for _, item in group.items()] == [1, 2, 3, 4, 5, 6]
    assert [item[0][1].label for _, item in group.items()] == \
        ['dataset1' for _ in range(3)] + ['dataset2' for _ in range(3)]

    result = list(calculate_group(group, model, parameter, data))
    assert len(result) == 6
    print(result[0])
    assert result[0][1].shape == (2, 2)
    assert result[3][1].shape == (2, 3)

    data = create_data_group(model, group, data)
    assert len(data) == 6
    assert data[0].shape[0] == 2
    assert data[3].shape[0] == 3

    data = {
        'dataset1': MockDataset([1, 2, 3], [5, 7]),
        'dataset2': MockDataset([4, 5, 6], [5, 7, 9]),
    }

    group = create_group(model, data, dataset='dataset1')
    assert len(group) == 3
    assert [item[0][0] for _, item in group.items()] == [1, 2, 3]
    assert all([item[0][1].label == 'dataset1' for _, item in group.items()])


def test_multi_dataset_overlap():
    model = MockModel.from_dict({
        "compartment": ["s1", "s2"],
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

    data = {
        'dataset1': MockDataset([0, 1, 2, 3], [5, 7]),
        'dataset2': MockDataset([1.4, 2.4, 3.4, 9], [5, 7, 9, 12]),
    }

    group = create_group(model, data)
    assert len(group) == 5
    assert group[0][0][1].label == 'dataset1'
    assert group[1][0][1].label == 'dataset1'
    assert group[1][1][1].label == 'dataset2'
    assert group[9][0][1].label == 'dataset2'

    result = list(calculate_group(group, model, parameter, data))
    assert len(result) == 5
    print(result[0])
    assert result[0][1].shape == (2, 2)
    assert result[1][1].shape == (2, 6)
    assert result[4][1].shape == (2, 4)

    data = create_data_group(model, group, data)
    assert len(data) == 5
    assert data[0].shape[0] == 2
    assert data[1].shape[0] == 6
    assert data[4].shape[0] == 4
