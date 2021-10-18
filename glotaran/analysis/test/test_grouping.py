import numpy as np
import xarray as xr

from glotaran.analysis.optimization_group import OptimizationGroup
from glotaran.analysis.test.models import SimpleTestModel
from glotaran.parameter import ParameterGroup
from glotaran.project import Scheme


def test_single_dataset():
    model = SimpleTestModel.from_dict(
        {
            "megacomplex": {"m1": {"is_index_dependent": False}},
            "dataset_groups": {"default": {"link_clp": True}},
            "dataset": {
                "dataset1": {
                    "megacomplex": ["m1"],
                },
            },
        }
    )
    print(model.validate())
    assert model.valid()

    parameters = ParameterGroup.from_list([1, 10])
    print(model.validate(parameters))
    assert model.valid(parameters)
    global_axis = [1, 2, 3]
    model_axis = [5, 7, 9, 12]

    data = {
        "dataset1": xr.DataArray(
            np.ones((3, 4)), coords=[("global", global_axis), ("model", model_axis)]
        ).to_dataset(name="data")
    }

    scheme = Scheme(model, parameters, data)
    optimization_group = OptimizationGroup(scheme, model.get_dataset_groups()["default"])
    bag = optimization_group._calculator.bag
    datasets = optimization_group._calculator.groups
    assert len(datasets) == 1
    assert len(bag) == 3
    assert all(p.data.size == 4 for p in bag)
    assert all(p.dataset_models[0].label == "dataset1" for p in bag)
    assert all(all(p.dataset_models[0].axis["model"] == model_axis) for p in bag)
    assert all(all(p.dataset_models[0].axis["global"] == global_axis) for p in bag)
    assert [p.dataset_models[0].indices["global"] for p in bag] == [0, 1, 2]


def test_multi_dataset_no_overlap():
    model = SimpleTestModel.from_dict(
        {
            "megacomplex": {"m1": {"is_index_dependent": False}},
            "dataset_groups": {"default": {"link_clp": True}},
            "dataset": {
                "dataset1": {
                    "megacomplex": ["m1"],
                },
                "dataset2": {
                    "megacomplex": ["m1"],
                },
            },
        }
    )

    model.grouped = lambda: True
    print(model.validate())
    assert model.valid()
    assert model.grouped()

    parameters = ParameterGroup.from_list([1, 10])
    print(model.validate(parameters))
    assert model.valid(parameters)

    global_axis_1 = [1, 2, 3]
    model_axis_1 = [5, 7]
    global_axis_2 = [4, 5, 6]
    model_axis_2 = [5, 7, 9]
    data = {
        "dataset1": xr.DataArray(
            np.ones((3, 2)), coords=[("global", global_axis_1), ("model", model_axis_1)]
        ).to_dataset(name="data"),
        "dataset2": xr.DataArray(
            np.ones((3, 3)), coords=[("global", global_axis_2), ("model", model_axis_2)]
        ).to_dataset(name="data"),
    }

    scheme = Scheme(model, parameters, data)
    optimization_group = OptimizationGroup(scheme, model.get_dataset_groups()["default"])
    bag = list(optimization_group._calculator.bag)
    assert len(optimization_group._calculator.groups) == 2
    assert len(bag) == 6
    assert all(p.data.size == 2 for p in bag[:3])
    assert all(p.dataset_models[0].label == "dataset1" for p in bag[:3])
    assert all(all(p.dataset_models[0].axis["model"] == model_axis_1) for p in bag[:3])
    assert all(all(p.dataset_models[0].axis["global"] == global_axis_1) for p in bag[:3])
    assert [p.dataset_models[0].indices["global"] for p in bag[:3]] == [0, 1, 2]

    assert all(p.data.size == 3 for p in bag[3:])
    assert all(p.dataset_models[0].label == "dataset2" for p in bag[3:])
    assert all(all(p.dataset_models[0].axis["model"] == model_axis_2) for p in bag[3:])
    assert all(all(p.dataset_models[0].axis["global"] == global_axis_2) for p in bag[3:])
    assert [p.dataset_models[0].indices["global"] for p in bag[3:]] == [0, 1, 2]


def test_multi_dataset_overlap():
    model = SimpleTestModel.from_dict(
        {
            "megacomplex": {"m1": {"is_index_dependent": False}},
            "dataset_groups": {"default": {"link_clp": True}},
            "dataset": {
                "dataset1": {
                    "megacomplex": ["m1"],
                },
                "dataset2": {
                    "megacomplex": ["m1"],
                },
            },
        }
    )

    model.grouped = lambda: True
    print(model.validate())
    assert model.valid()
    assert model.grouped()

    parameters = ParameterGroup.from_list([1, 10])
    print(model.validate(parameters))
    assert model.valid(parameters)

    global_axis_1 = [1, 2, 3, 5]
    model_axis_1 = [5, 7]
    global_axis_2 = [0, 1.4, 2.4, 3.4, 9]
    model_axis_2 = [5, 7, 9, 12]
    data = {
        "dataset1": xr.DataArray(
            np.ones((4, 2)), coords=[("global", global_axis_1), ("model", model_axis_1)]
        ).to_dataset(name="data"),
        "dataset2": xr.DataArray(
            np.ones((5, 4)), coords=[("global", global_axis_2), ("model", model_axis_2)]
        ).to_dataset(name="data"),
    }

    scheme = Scheme(model, parameters, data, clp_link_tolerance=5e-1)
    optimization_group = OptimizationGroup(scheme, model.get_dataset_groups()["default"])
    bag = list(optimization_group._calculator.bag)
    assert len(optimization_group._calculator.groups) == 3
    assert "dataset1dataset2" in optimization_group._calculator.groups
    assert optimization_group._calculator.groups["dataset1dataset2"] == ["dataset1", "dataset2"]
    assert len(bag) == 6

    assert all(p.data.size == 4 for p in bag[:1])
    assert all(p.dataset_models[0].label == "dataset1" for p in bag[1:5])
    assert all(all(p.dataset_models[0].axis["model"] == model_axis_1) for p in bag[1:5])
    assert all(all(p.dataset_models[0].axis["global"] == global_axis_1) for p in bag[1:5])
    assert [p.dataset_models[0].indices["global"] for p in bag[1:5]] == [0, 1, 2, 3]

    assert all(p.data.size == 6 for p in bag[1:4])
    assert all(p.dataset_models[1].label == "dataset2" for p in bag[1:4])
    assert all(all(p.dataset_models[1].axis["model"] == model_axis_2) for p in bag[1:4])
    assert all(all(p.dataset_models[1].axis["global"] == global_axis_2) for p in bag[1:4])
    assert [p.dataset_models[1].indices["global"] for p in bag[1:4]] == [1, 2, 3]

    assert all(p.data.size == 4 for p in bag[5:])
    assert bag[4].dataset_models[0].label == "dataset1"
    assert bag[5].dataset_models[0].label == "dataset2"
    assert np.array_equal(bag[4].dataset_models[0].axis["model"], model_axis_1)
    assert np.array_equal(bag[5].dataset_models[0].axis["model"], model_axis_2)
    assert [p.dataset_models[0].indices["global"] for p in bag[1:4]] == [0, 1, 2]
