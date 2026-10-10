from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.builtin.elements.baseline import BaselineElement
from glotaran.model.data_model import DataModel
from glotaran.optimization import OptimizationData
from glotaran.optimization import OptimizationMatrix


def test_baseline():
    model = DataModel(
        data=xr.DataArray(np.ones((1, 1)), coords=[("model", [0]), ("global", [0])]).to_dataset(
            name="data"
        ),
        elements=[BaselineElement(type="baseline", label="test", dimension="model")],
    )
    data = OptimizationData(model)

    matrix = OptimizationMatrix.from_data(data)

    assert len(matrix.clp_axis) == 1
    assert ("baseline_test") in matrix.clp_axis

    assert matrix.array.shape == (data.model_axis.size, 1)
    assert np.all(matrix.array[:, 0] == 1)
