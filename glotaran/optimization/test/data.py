from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.model import DataModel
from glotaran.optimization.test.elements import TestElementConstant

TestDataModelConstantIndexIndependent = DataModel(
    data=xr.DataArray(
        np.ones((4, 3)) * 5, coords=[("model", [5, 7, 9, 12]), ("global", [1, 5, 6])]
    ).to_dataset(name="data"),
    elements=[
        TestElementConstant(
            type="test-element-constant",
            label="test",
            dimension="model",
            compartments=["c1"],
            value=5,
            is_index_dependent=False,
        )
    ],
)

TestDataModelConstantIndexDependent = DataModel(
    data=xr.DataArray(
        np.ones((4, 3)) * 2, coords=[("global", [0, 3, 7, 10]), ("model", [4, 11, 15])]
    ).to_dataset(name="data"),
    elements=[
        TestElementConstant(
            type="test-element-constant",
            label="test",
            dimension="model",
            compartments=["c2"],
            value=2,
            is_index_dependent=True,
        )
    ],
)
TestDataModelConstantThreeCompartments = DataModel(
    data=xr.DataArray(
        np.ones((6, 5)) * 3, coords=[("global", [1, 3, 7, 8, 9, 10]), ("model", [4, 5, 7, 11, 15])]
    ).to_dataset(name="data"),
    elements=[
        TestElementConstant(
            type="test-element-constant",
            label="test",
            dimension="model",
            compartments=[
                "c3_1",
                "c3_2",
                "c3_3",
            ],
            value=3,
            is_index_dependent=False,
        )
    ],
)

TestDataModelGlobal = DataModel(
    data=xr.DataArray(
        np.ones((3, 4)) * 2, coords=[("model", [4, 11, 15]), ("global", [0, 3, 7, 10])]
    ).to_dataset(name="data"),
    elements=[
        TestElementConstant(
            type="test-element-constant",
            label="test",
            dimension="model",
            compartments=["c4"],
            value=5,
            is_index_dependent=False,
        )
    ],
    element_scale={"test": 1},
    global_elements=[
        TestElementConstant(
            type="test-element-constant",
            label="test_global",
            dimension="global",
            compartments=["c4"],
            value=4,
            is_index_dependent=False,
        )
    ],
    global_element_scale={"test_global": 1},
)
