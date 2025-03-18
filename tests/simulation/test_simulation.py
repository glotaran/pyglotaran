from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from glotaran.simulation import simulate
from tests.optimization.data import TestDataModelConstantIndexDependent
from tests.optimization.data import TestDataModelConstantIndexIndependent
from tests.optimization.data import TestDataModelGlobal


@pytest.mark.parametrize("is_index_dependent", [True, False])
@pytest.mark.parametrize("noise", [True, False])
def test_simulate_from_clp(is_index_dependent: bool, noise: bool):
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
            ("global_dim", global_axis),
            ("clp_label", [f"c{2 if is_index_dependent else 1}"]),
        ),
    )

    data = simulate(
        model,
        {},
        None,
        {"global_dim": global_axis, "model_dim": model_axis},
        clp=clp,
        noise=noise,
        noise_std_dev=0.1,
    )
    assert np.array_equal(data["global_dim"], global_axis)
    assert np.array_equal(data["model_dim"], model_axis)
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
    data = simulate(model, {}, None, {"global_dim": global_axis, "model_dim": model_axis})
    assert np.array_equal(data["global_dim"], global_axis)
    assert np.array_equal(data["model_dim"], model_axis)
    assert data.data.shape == (3, 4)
