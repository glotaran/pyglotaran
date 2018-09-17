
import numpy as np

from glotaran.analysis.simulation import simulate
from glotaran.model import ParameterGroup

from .mock import MockModel


def test_simulate_dataset():
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

    parameter = ParameterGroup.from_list([1, 1])
    print(model.errors_parameter(parameter))
    assert model.valid_parameter(parameter)

    est_axis = np.asarray([1, 1, 1, 1])
    cal_axis = np.asarray([2, 2, 2])

    data = simulate(model, parameter, 'dataset1', {'e': est_axis, 'c': cal_axis})
    assert np.array_equal(data.get_axis("c"), cal_axis)
    assert np.array_equal(data.get_axis("e"), est_axis)
    assert data.data().shape == (4, 3)
    assert np.array_equal(data.data(), np.asarray([
        [2, 4, 6],
        [4, 10, 16],
        [6, 16, 26],
        [8, 22, 36],
    ]))
