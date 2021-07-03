import numpy as np
import pytest

from glotaran.analysis.simulation import simulate
from glotaran.analysis.test.models import SimpleTestModel
from glotaran.parameter import ParameterGroup


@pytest.mark.parametrize("index_dependent", [True, False])
@pytest.mark.parametrize("noise", [True, False])
def test_simulate_dataset(index_dependent, noise):
    model = SimpleTestModel.from_dict(
        {
            "megacomplex": {
                "m1": {"is_index_dependent": index_dependent},
                "m2": {"type": "global_complex"},
            },
            "dataset": {
                "dataset1": {
                    "megacomplex": ["m1"],
                    "global_megacomplex": ["m2"],
                },
            },
        }
    )
    print(model.validate())
    assert model.valid()

    parameter = ParameterGroup.from_list([1, 1])
    print(model.validate(parameter))
    assert model.valid(parameter)

    global_axis = np.asarray([1, 1, 1, 1])
    model_axis = np.asarray([2, 2, 2])

    data = simulate(
        model,
        "dataset1",
        parameter,
        {"global": global_axis, "model": model_axis},
        noise=noise,
        noise_std_dev=0.1,
    )
    assert np.array_equal(data["global"], global_axis)
    assert np.array_equal(data["model"], model_axis)
    assert data.data.shape == (3, 4)
    if not noise:
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
