import pytest
import xarray as xr

from glotaran.analysis.scheme import Scheme
from .mock import MockModel # noqa


@pytest.fixture(scope='session')
def scheme(tmpdir_factory):

    path = tmpdir_factory.mktemp("scheme")

    model = "type: mock\ndataset:\n  dataset1:\n    megacomplex: []"
    model_path = path.join("model.yml")
    with open(model_path, 'w') as f:
        f.write(model)

    parameter = "[1.0, 67.0]"
    parameter_path = path.join("parameter.yml")
    with open(parameter_path, 'w') as f:
        f.write(parameter)

    dataset_path = path.join('dataset.nc')
    xr.DataArray([1, 2, 3]).to_dataset(name='data').to_netcdf(dataset_path)

    scheme = f"""
    model: {model_path}
    parameter: {parameter_path}
    nnls: True
    nfev: 42
    data:
      dataset1: {dataset_path}
    """
    scheme_path = path.join('scheme.gta')
    with open(scheme_path, 'w') as f:
        f.write(scheme)
    return scheme_path


def test_scheme(scheme):
    scheme = Scheme.from_yml_file(scheme)
    assert scheme.model is not None
    assert scheme.model.model_type == "mock"

    assert scheme.parameter is not None
    assert scheme.parameter.get("1") == 1.0
    assert scheme.parameter.get("2") == 67.0

    assert scheme.nnls
    assert scheme.nfev == 42

    assert 'dataset1' in scheme.data
    assert scheme.data['dataset1'].data.size == 3
