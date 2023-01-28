import numpy as np
import pytest
import xarray as xr

from glotaran.model import ClpRelation
from glotaran.model import DataModel
from glotaran.model import ZeroConstraint
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.optimization.test.models import TestMegacomplexConstant


def data_model_one() -> DataModel:
    global_axis = [1, 5, 6]
    model_axis = [5, 7, 9, 12]

    data = xr.DataArray(
        np.ones((4, 3)), coords=[("model", model_axis), ("global", global_axis)]
    ).to_dataset(name="data")

    return DataModel(
        data=data,
        megacomplex=[
            TestMegacomplexConstant(
                type="test-megacomplex-constant",
                label="test",
                dimension="model",
                compartments=["a"],
                value=5,
                is_index_dependent=False,
            )
        ],
    )


def data_model_two() -> DataModel:
    global_axis = [0, 3, 7, 10]
    model_axis = [4, 11, 15]

    data = xr.DataArray(
        np.ones((4, 3)) * 2, coords=[("global", global_axis), ("model", model_axis)]
    ).to_dataset(name="data")
    return DataModel(
        data=data,
        megacomplex=[
            TestMegacomplexConstant(
                type="test-megacomplex-constant",
                label="test",
                dimension="model",
                compartments=["b"],
                value=3,
                is_index_dependent=True,
            )
        ],
    )


def data_model_three() -> DataModel:
    global_axis = [0, 3, 7, 10]
    model_axis = [4, 11, 15]

    data = xr.DataArray(
        np.ones((4, 3)) * 2, coords=[("global", global_axis), ("model", model_axis)]
    ).to_dataset(name="data")
    return DataModel(
        data=data,
        megacomplex=[
            TestMegacomplexConstant(
                type="test-megacomplex-constant",
                label="test",
                dimension="model",
                compartments=["a", "b", "c"],
                value=3,
                is_index_dependent=False,
            )
        ],
    )


@pytest.mark.parametrize("weight", (True, False))
@pytest.mark.parametrize("data_model", (data_model_one, data_model_two))
def test_from_data(weight, data_model):
    data_model = data_model()
    if weight:
        data_model.data["weight"] = xr.ones_like(data_model.data.data) * 0.5
    data = OptimizationData(data_model)
    matrix = OptimizationMatrix.from_data(data)
    assert matrix.array.shape == (
        (data.data.shape[1], data.data.shape[0], 1)
        if weight or data_model.megacomplex[0].is_index_dependent
        else (data.data.shape[0], 1)
    )
    assert matrix.clp_labels == (["b"] if data_model.megacomplex[0].is_index_dependent else ["a"])


def test_constraints():

    constraints = [ZeroConstraint(type="zero", target="c", interval=[(3, 7)])]
    data = OptimizationData(data_model_three())
    matrix = OptimizationMatrix.from_data(data)
    assert matrix.array.shape == (3, 3)
    assert matrix.clp_labels == ["a", "b", "c"]
    reduced_matrix = matrix.reduce(0, constraints, [])
    assert reduced_matrix.array.shape == (3, 3)
    assert reduced_matrix.clp_labels == ["a", "b", "c"]
    reduced_matrix = matrix.reduce(3, constraints, [])
    assert reduced_matrix.array.shape == (3, 2)
    assert reduced_matrix.clp_labels == ["a", "b"]
