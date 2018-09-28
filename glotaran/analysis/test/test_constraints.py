import numpy as np
from glotaran.analysis.grouping import apply_constraints
from glotaran.model import ParameterGroup

from .mock import MockModel


def test_zero_and_only():
    model = MockModel.from_dict({
        "compartment": ["s1", "s2"],
        "initial_concentration": {
            "j1": [["1", "1"]]
        },
        "dataset": {
            "dataset1": {
                "initial_concentration": 'j1',
                "megacomplex": [],
                "compartment_constraints": [
                    {'type': 'only', 'compartment': 's1', 'interval': [(1, 2)]},
                    {'type': 'zero', 'compartment': 's2', 'interval': [(2, 3)]},
                ]
            },
        }
    })
    print(model.errors())
    assert model.valid()

    parameter = ParameterGroup.from_list([1])
    print(model.errors_parameter(parameter))
    assert model.valid_parameter(parameter)

    matrix = np.asarray([[1, 1], [1, 1]])
    apply_constraints(model.dataset['dataset1'], model.compartment, matrix, 1)
    assert np.all(matrix[0] == 1)
    assert np.all(matrix[1] == 1)

    matrix = np.asarray([[1, 1], [1, 1]])
    apply_constraints(model.dataset['dataset1'], model.compartment, matrix, 2)
    assert np.all(matrix[0] == 1)
    assert np.all(matrix[1] == 0)

    matrix = np.asarray([[1, 1], [1, 1]])
    apply_constraints(model.dataset['dataset1'], model.compartment, matrix, 3)
    assert np.all(matrix[0] == 0)
    assert np.all(matrix[1] == 0)


def test_equal():
    model = MockModel.from_dict({
        "compartment": ["s1", "s2", "s3", "s4"],
        "initial_concentration": {
            "j1": [["1", "1", "1", "1"]]
        },
        "dataset": {
            "dataset1": {
                "initial_concentration": 'j1',
                "megacomplex": [],
                "compartment_constraints": [
                    {'type': 'equal',
                     'compartment': 's1',
                     'interval': [(1, 2)],
                     'targets': {'s3': '2'},
                     },
                    {'type': 'equal',
                     'compartment': 's2',
                     'interval': [(1, 2)],
                     'targets': {'s3': '3', "s4": '4'},
                     },
                ]
            },
        }
    })
    print(model.errors())
    assert model.valid()

    parameter = ParameterGroup.from_list([1, 3, 0.5, 2])
    print(model.errors_parameter(parameter))
    assert model.valid_parameter(parameter)

    matrix = np.asarray([
        [1, 1],
        [1, 1],
        [76, 76],
        [2, 2],
    ], dtype=np.float64)
    apply_constraints(model.dataset['dataset1'].fill(model, parameter),
                      model.compartment, matrix, 1)
    assert np.all(matrix[0] == 228)
    assert np.allclose(matrix[1], 42)
