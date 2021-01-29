import numpy as np
import xarray as xr

from glotaran.analysis.problem import Problem
from glotaran.analysis.scheme import Scheme
from glotaran.analysis.test.models import SimpleTestModel
from glotaran.parameter import ParameterGroup


def test_single_dataset():
    model = SimpleTestModel.from_dict(
        {
            "dataset": {
                "dataset1": {
                    "megacomplex": [],
                },
            }
        }
    )
    model.grouped = lambda: True
    print(model.validate())
    assert model.valid()
    assert model.grouped()

    parameters = ParameterGroup.from_list([1, 10])
    print(model.validate(parameters))
    assert model.valid(parameters)
    axis_e = [1, 2, 3]
    axis_c = [5, 7, 9, 12]

    data = {
        "dataset1": xr.DataArray(
            np.ones((3, 4)), coords=[("e", axis_e), ("c", axis_c)]
        ).to_dataset(name="data")
    }

    scheme = Scheme(model, parameters, data)
    problem = Problem(scheme)
    bag = problem.bag
    datasets = problem.groups
    assert len(datasets) == 1
    assert len(bag) == 3
    assert all(p.data.size == 4 for p in bag)
    assert all(p.descriptor[0].label == "dataset1" for p in bag)
    assert all(all(p.descriptor[0].axis == axis_c) for p in bag)
    assert [p.descriptor[0].index for p in bag] == axis_e


def test_multi_dataset_no_overlap():
    model = SimpleTestModel.from_dict(
        {
            "dataset": {
                "dataset1": {
                    "megacomplex": [],
                },
                "dataset2": {
                    "megacomplex": [],
                },
            }
        }
    )

    model.grouped = lambda: True
    print(model.validate())
    assert model.valid()
    assert model.grouped()

    parameters = ParameterGroup.from_list([1, 10])
    print(model.validate(parameters))
    assert model.valid(parameters)

    axis_e_1 = [1, 2, 3]
    axis_c_1 = [5, 7]
    axis_e_2 = [4, 5, 6]
    axis_c_2 = [5, 7, 9]
    data = {
        "dataset1": xr.DataArray(
            np.ones((3, 2)), coords=[("e", axis_e_1), ("c", axis_c_1)]
        ).to_dataset(name="data"),
        "dataset2": xr.DataArray(
            np.ones((3, 3)), coords=[("e", axis_e_2), ("c", axis_c_2)]
        ).to_dataset(name="data"),
    }

    scheme = Scheme(model, parameters, data)
    problem = Problem(scheme)
    bag = list(problem.bag)
    assert len(problem.groups) == 2
    assert len(bag) == 6
    assert all(p.data.size == 2 for p in bag[:3])
    assert all(p.descriptor[0].label == "dataset1" for p in bag[:3])
    assert all(all(p.descriptor[0].axis == axis_c_1) for p in bag[:3])
    assert [p.descriptor[0].index for p in bag[:3]] == axis_e_1

    assert all(p.data.size == 3 for p in bag[3:])
    assert all(p.descriptor[0].label == "dataset2" for p in bag[3:])
    assert all(all(p.descriptor[0].axis == axis_c_2) for p in bag[3:])
    assert [p.descriptor[0].index for p in bag[3:]] == axis_e_2


def test_multi_dataset_overlap():
    model = SimpleTestModel.from_dict(
        {
            "dataset": {
                "dataset1": {
                    "megacomplex": [],
                },
                "dataset2": {
                    "megacomplex": [],
                },
            }
        }
    )

    model.grouped = lambda: True
    print(model.validate())
    assert model.valid()
    assert model.grouped()

    parameters = ParameterGroup.from_list([1, 10])
    print(model.validate(parameters))
    assert model.valid(parameters)

    axis_e_1 = [1, 2, 3, 5]
    axis_c_1 = [5, 7]
    axis_e_2 = [0, 1.4, 2.4, 3.4, 9]
    axis_c_2 = [5, 7, 9, 12]
    data = {
        "dataset1": xr.DataArray(
            np.ones((4, 2)), coords=[("e", axis_e_1), ("c", axis_c_1)]
        ).to_dataset(name="data"),
        "dataset2": xr.DataArray(
            np.ones((5, 4)), coords=[("e", axis_e_2), ("c", axis_c_2)]
        ).to_dataset(name="data"),
    }

    scheme = Scheme(model, parameters, data, group_tolerance=5e-1)
    problem = Problem(scheme)
    bag = list(problem.bag)
    assert len(problem.groups) == 3
    assert "dataset1dataset2" in problem.groups
    assert problem.groups["dataset1dataset2"] == ["dataset1", "dataset2"]
    assert len(bag) == 6

    assert all(p.data.size == 4 for p in bag[:1])
    assert all(p.descriptor[0].label == "dataset1" for p in bag[1:5])
    assert all(all(p.descriptor[0].axis == axis_c_1) for p in bag[1:5])
    assert [p.descriptor[0].index for p in bag[1:5]] == axis_e_1

    assert all(p.data.size == 6 for p in bag[1:4])
    assert all(p.descriptor[1].label == "dataset2" for p in bag[1:4])
    assert all(all(p.descriptor[1].axis == axis_c_2) for p in bag[1:4])
    assert [p.descriptor[1].index for p in bag[1:4]] == axis_e_2[1:4]

    assert all(p.data.size == 4 for p in bag[5:])
    assert bag[4].descriptor[0].label == "dataset1"
    assert bag[5].descriptor[0].label == "dataset2"
    assert np.array_equal(bag[4].descriptor[0].axis, axis_c_1)
    assert np.array_equal(bag[5].descriptor[0].axis, axis_c_2)
    assert [p.descriptor[0].index for p in bag[1:4]] == axis_e_1[:-1]
