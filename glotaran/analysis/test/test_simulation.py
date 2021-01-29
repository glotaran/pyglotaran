import numpy as np

from glotaran.analysis.simulation import simulate
from glotaran.parameter import ParameterGroup

from .models import SimpleTestModel


def test_simulate_dataset():
    model = SimpleTestModel.from_dict(
        {
            "dataset": {
                "dataset1": {
                    "megacomplex": [],
                },
            }
        }
    )
    print(model.validate())
    assert model.valid()

    parameter = ParameterGroup.from_list([1, 1])
    print(model.validate(parameter))
    assert model.valid(parameter)

    est_axis = np.asarray([1, 1, 1, 1])
    cal_axis = np.asarray([2, 2, 2])

    data = simulate(model, "dataset1", parameter, {"e": est_axis, "c": cal_axis})
    assert np.array_equal(data["c"], cal_axis)
    assert np.array_equal(data["e"], est_axis)
    assert data.data.shape == (3, 4)
    assert np.array_equal(
        data.data,
        np.asarray(
            [
                [2, 4, 6],
                [4, 10, 16],
                [6, 16, 26],
                [8, 22, 36],
            ]
        ).T,
    )
