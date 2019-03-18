import numpy as np
import xarray as xr

from glotaran.analysis.optimizer import Optimizer
from glotaran.analysis.scheme import Scheme
from glotaran.parameter import ParameterGroup

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

    data = {'dataset1': xr.DataArray(
        np.ones((3, 4)),
        coords=[('e', [1, 2, 3]), ('c', [5, 7, 9, 12])]
    ).to_dataset(name="data")}

    scheme = Scheme(model, parameter, data)
    optimizer = Optimizer(scheme)
    group = optimizer._global_problem
    assert len(group) == 3
    assert list(group.keys()) == [f"dataset1_{i}" for i in [1, 2, 3]]
    assert all([p.dataset_descriptor.label == 'dataset1' for p in group.values()])

    optimizer._create_calculate_penalty_job(parameter)
    result = [m.compute() for m in optimizer.matrices.values()]
    assert len(result) == 3
    print(result[0])
    assert result[0].shape == (4, 2)

    data = optimizer._global_data
    assert len(data) == 3
    assert list(data.values())[1].compute().shape[0] == 4


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

    data = {
        'dataset1': xr.DataArray(
            np.ones((3, 2)),
            coords=[('e', [1, 2, 3]), ('c', [5, 7])]
        ).to_dataset(name="data"),
        'dataset2': xr.DataArray(
            np.ones((3, 3)),
            coords=[('e', [4, 5, 6]), ('c', [5, 7, 9])]
        ).to_dataset(name="data"),
    }

    scheme = Scheme(model, parameter, data)
    optimizer = Optimizer(scheme)
    group = optimizer._global_problem
    assert len(group) == 6
    assert [problem[0][0] for problem in group.values()] == [1, 2, 3, 4, 5, 6]
    assert [problem[0][1].label for problem in group.values()] == \
        ['dataset1' for _ in range(3)] + ['dataset2' for _ in range(3)]

    optimizer._create_calculate_penalty_job(parameter)
    result = [m.compute() for mat in optimizer.matrices.values() for m in mat]
    assert len(result) == 6
    print(result[0])
    assert result[0].shape == (2, 2)
    assert result[3].shape == (3, 2)

    data = optimizer._global_data
    assert len(data) == 6
    assert list(data.values())[1].compute().shape[0] == 2
    assert list(data.values())[4].compute().shape[0] == 3


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

    data = {
        'dataset1': xr.DataArray(
            np.ones((4, 2)),
            coords=[('e', [0, 1, 2, 3]), ('c', [5, 7])]
        ).to_dataset(name="data"),
        'dataset2': xr.DataArray(
            np.ones((4, 4)),
            coords=[('e', [1.4, 2.4, 3.4, 9]), ('c', [5, 7, 9, 12])]
        ).to_dataset(name="data"),
    }

    scheme = Scheme(model, parameter, data, group_tolerance=5e-1)
    optimizer = Optimizer(scheme)
    group = optimizer._global_problem
    assert len(group) == 5
    assert group[0][0][1].label == 'dataset1'
    assert group[1][0][1].label == 'dataset1'
    assert group[1][1][1].label == 'dataset2'
    assert group[9][0][1].label == 'dataset2'

    optimizer._create_calculate_penalty_job(parameter)
    print(optimizer.matrices)
    result = [m.compute() for m in optimizer.full_matrices.values()]
    assert len(result) == 5
    print(result[0])
    print(result[1])
    assert result[0].shape == (2, 2)
    assert result[1].shape == (6, 2)
    assert result[4].shape == (4, 2)

    data = [d.compute() for d in optimizer._global_data.values()]
    assert len(data) == 5
    assert data[0].shape[0] == 2
    assert data[1].shape[0] == 6
    assert data[4].shape[0] == 4
