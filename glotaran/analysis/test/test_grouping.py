import numpy as np
import xarray as xr

from glotaran.analysis.scheme import Scheme
from glotaran.parameter import ParameterGroup

from glotaran.analysis.problem_bag import create_grouped_bag


from .mock import MockModel


def test_single_dataset():
    model = MockModel.from_dict({
        "dataset": {
            "dataset1": {
                "megacomplex": [],
            },
        }
    })
    print(model.validate())
    assert model.valid()

    parameter = ParameterGroup.from_list([1, 10])
    print(model.validate(parameter))
    assert model.valid(parameter)
    axis_e = [1, 2, 3]
    axis_c = [5, 7, 9, 12]

    data = {'dataset1': xr.DataArray(
        np.ones((3, 4)),
        coords=[('e', axis_e), ('c', axis_c)]
    ).to_dataset(name="data")}

    scheme = Scheme(model, parameter, data)
    bag, datasets = create_grouped_bag(scheme)
    bag = bag.compute()
    assert len(datasets) == 0
    assert len(bag) == 3
    assert all([p.data.size == 4 for p in bag])
    assert all([p.descriptor[0].dataset == 'dataset1' for p in bag])
    assert all([all(p.descriptor[0].axis == axis_c) for p in bag])
    assert [p.descriptor[0].index for p in bag] == axis_e


def test_multi_dataset_no_overlap():
    model = MockModel.from_dict({
        "dataset": {
            "dataset1": {
                "megacomplex": [],
            },
            "dataset2": {
                "megacomplex": [],
            },
        }
    })

    print(model.validate())
    assert model.valid()

    parameter = ParameterGroup.from_list([1, 10])
    print(model.validate(parameter))
    assert model.valid(parameter)

    axis_e_1 = [1, 2, 3]
    axis_c_1 = [5, 7]
    axis_e_2 = [4, 5, 6]
    axis_c_2 = [5, 7, 9]
    data = {
        'dataset1': xr.DataArray(
            np.ones((3, 2)),
            coords=[('e', axis_e_1), ('c', axis_c_1)]
        ).to_dataset(name="data"),
        'dataset2': xr.DataArray(
            np.ones((3, 3)),
            coords=[('e', axis_e_2), ('c', axis_c_2)]
        ).to_dataset(name="data"),
    }

    scheme = Scheme(model, parameter, data)
    bag, datasets = create_grouped_bag(scheme)
    bag = bag.compute()
    assert len(datasets) == 0
    assert len(bag) == 6
    assert all([p.data.size == 2 for p in bag[:3]])
    assert all([p.descriptor[0].dataset == 'dataset1' for p in bag[:3]])
    assert all([all(p.descriptor[0].axis == axis_c_1) for p in bag[:3]])
    assert [p.descriptor[0].index for p in bag[:3]] == axis_e_1

    assert all([p.data.size == 3 for p in bag[3:]])
    assert all([p.descriptor[0].dataset == 'dataset2' for p in bag[3:]])
    assert all([all(p.descriptor[0].axis == axis_c_2) for p in bag[3:]])
    assert [p.descriptor[0].index for p in bag[3:]] == axis_e_2


def test_multi_dataset_overlap():
    model = MockModel.from_dict({
        "dataset": {
            "dataset1": {
                "megacomplex": [],
            },
            "dataset2": {
                "megacomplex": [],
            },
        }
    })

    print(model.validate())
    assert model.valid()

    parameter = ParameterGroup.from_list([1, 10])
    print(model.validate(parameter))
    assert model.valid(parameter)

    axis_e_1 = [0, 1, 2, 3]
    axis_c_1 = [5, 7]
    axis_e_2 = [1.4, 2.4, 3.4, 9]
    axis_c_2 = [5, 7, 9, 12]
    data = {
        'dataset1': xr.DataArray(
            np.ones((4, 2)),
            coords=[('e', axis_e_1), ('c', axis_c_1)]
        ).to_dataset(name="data"),
        'dataset2': xr.DataArray(
            np.ones((4, 4)),
            coords=[('e', axis_e_2), ('c', axis_c_2)]
        ).to_dataset(name="data"),
    }

    scheme = Scheme(model, parameter, data, group_tolerance=5e-1)
    bag, datasets = create_grouped_bag(scheme)
    bag = bag.compute()
    assert len(datasets) == 1
    assert "dataset1dataset2" in datasets
    assert datasets['dataset1dataset2'] == ["dataset1", "dataset2"]
    assert len(bag) == 5

    assert all([p.data.size == 2 for p in bag[:1]])
    assert all([p.descriptor[0].dataset == 'dataset1' for p in bag[:4]])
    assert all([all(p.descriptor[0].axis == axis_c_1) for p in bag[:4]])
    assert [p.descriptor[0].index for p in bag[:4]] == axis_e_1

    assert all([p.data.size == 6 for p in bag[1:4]])
    assert all([p.descriptor[1].dataset == 'dataset2' for p in bag[1:4]])
    assert all([all(p.descriptor[1].axis == axis_c_2) for p in bag[1:4]])
    assert [p.descriptor[1].index for p in bag[1:4]] == axis_e_2[:3]

    assert all([p.data.size == 4 for p in bag[4:]])
    assert all([p.descriptor[0].dataset == 'dataset2' for p in bag[4:]])
    assert all([all(p.descriptor[0].axis == axis_c_2) for p in bag[4:]])
    assert [p.descriptor[0].index for p in bag[4:]] == axis_e_2[3:]
