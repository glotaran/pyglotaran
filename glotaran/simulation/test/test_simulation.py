from copy import deepcopy

import numpy as np
import pytest

from glotaran.optimization.test.models import SimpleTestModel
from glotaran.optimization.test.suites import FullModel
from glotaran.parameter import Parameters
from glotaran.simulation import simulate


@pytest.mark.parametrize("index_dependent", [True, False])
@pytest.mark.parametrize("noise", [True, False])
def test_simulate_dataset(index_dependent, noise):
    model = SimpleTestModel(
        **{
            "megacomplex": {
                "m1": {"type": "simple-test-mc", "is_index_dependent": index_dependent},
                "m2": {"type": "simple-test-mc", "is_index_dependent": False},
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

    parameter = Parameters.from_list([1, 1])
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


def test_simulate_full_model_same_result_when_swapping_global_and_model_megacomplex():
    """Same result when swapping the global and model megacomplex of a dataset model.

    The difference originated from applying the noise, which operated on the raw
    transposed numpy array. This then lead to the noise being applied differently.
    """
    model = deepcopy(FullModel.model)
    swapped_model = deepcopy(FullModel.model)
    noise_seed = 42

    dataset_model = model.dataset["dataset1"]
    swapped_dataset_model = swapped_model.dataset["dataset1"]
    swapped_dataset_model.megacomplex = dataset_model.global_megacomplex
    swapped_dataset_model.global_megacomplex = dataset_model.megacomplex

    result = simulate(
        model,
        "dataset1",
        FullModel.parameters,
        FullModel.coordinates,
        noise=True,
        noise_std_dev=0.1,
        noise_seed=noise_seed,
    )
    swapped_result = simulate(
        swapped_model,
        "dataset1",
        FullModel.parameters,
        FullModel.coordinates,
        noise=True,
        noise_std_dev=0.1,
        noise_seed=noise_seed,
    )

    assert np.array_equal(result["global"], swapped_result["global"])
    assert np.array_equal(result["model"], swapped_result["model"])
    assert result.data.shape == swapped_result.data.T.shape
    np.testing.assert_allclose(result.data, swapped_result.data.T)
