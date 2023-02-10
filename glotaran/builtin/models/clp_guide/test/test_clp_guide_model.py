import numpy as np
import xarray as xr

from glotaran.builtin.models.clp_guide import ClpGuideModel
from glotaran.model import DataModel
from glotaran.optimization import OptimizationData
from glotaran.optimization import OptimizationMatrix


def test_clp_guide():
    model = DataModel(
        data=xr.DataArray(np.ones((1, 1)), coords=[("model", [0]), ("global", [0])]).to_dataset(
            name="data"
        ),
        models=[ClpGuideModel(type="clp-guide", label="test", target="c", dimension="model")],
    )
    data = OptimizationData(model)

    matrix = OptimizationMatrix.from_data(data)

    assert len(matrix.clp_axis) == 1
    assert ("c") in matrix.clp_axis

    assert matrix.array.shape == (1, 1)
    assert np.all(matrix.array[0, 0] == 1)
