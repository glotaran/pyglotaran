import numpy as np
import pytest
import xarray as xr

from glotaran.optimization.test.data import TestDataModelConstantIndexDependent
from glotaran.optimization.test.data import TestDataModelConstantIndexIndependent
from glotaran.optimization.test.data import TestDataModelGlobal
from glotaran.simulation import simulate


@pytest.mark.parametrize("is_index_dependent", [True, False])
@pytest.mark.parametrize("noise", [True, False])
def test_simulate_from_clp(is_index_dependent, noise):
    model = (
        TestDataModelConstantIndexDependent
        if is_index_dependent
        else TestDataModelConstantIndexIndependent
    )
    global_axis = np.asarray([1, 2, 3, 4])
    model_axis = np.asarray([1, 2, 3])

    clp = xr.DataArray(
        [[10], [20], [30], [40]],
        coords=(
            ("global", global_axis),
            ("clp_label", [f"c{2 if is_index_dependent else 1}"]),
        ),
    )

    data = simulate(
        model,
        {},
        None,
        {"global": global_axis, "model": model_axis},
        clp=clp,
        noise=noise,
        noise_std_dev=0.1,
    )
    assert np.array_equal(data["global"], global_axis)
    assert np.array_equal(data["model"], model_axis)
    assert data.data.shape == (3, 4)
    print(data.data)
    if not noise:
        if is_index_dependent:
            assert np.array_equal(
                data.data.data,
                np.array(
                    [
                        [20.0, 40.0, 60.0, 80.0],
                        [20.0, 40.0, 60.0, 80.0],
                        [20.0, 40.0, 60.0, 80.0],
                    ]
                ),
            )
        else:
            assert np.array_equal(
                data.data.data,
                np.array(
                    [
                        [50, 100, 150, 200],
                        [50, 100, 150, 200],
                        [50, 100, 150, 200],
                    ]
                ),
            )


def test_simulate_from_global_model():
    model = TestDataModelGlobal
    global_axis = np.asarray([1, 2, 3, 4])
    model_axis = np.asarray([1, 2, 3])
    data = simulate(model, {}, None, {"global": global_axis, "model": model_axis})
    assert np.array_equal(data["global"], global_axis)
    assert np.array_equal(data["model"], model_axis)
    assert data.data.shape == (3, 4)
