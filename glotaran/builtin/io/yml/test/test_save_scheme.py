from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from glotaran.examples.sequential import dataset
from glotaran.examples.sequential import model
from glotaran.examples.sequential import parameter
from glotaran.io import load_scheme
from glotaran.io import save_dataset
from glotaran.io import save_model
from glotaran.io import save_parameters
from glotaran.io import save_scheme
from glotaran.project import Scheme

if TYPE_CHECKING:
    from pathlib import Path


want = """add_svd: true
clp_link_tolerance: 0.0
data_files:
  dataset_1: d.nc
ftol: 1.0e-08
gtol: 1.0e-08
maximum_number_function_evaluations: null
model_file: m.yml
optimization_method: TrustRegionReflection
parameters_file: p.csv
result_path: null
xtol: 1.0e-08
"""


def test_save_scheme(tmp_path: Path):
    scheme = Scheme(
        model,
        parameter,
        {"dataset_1": dataset},
        model_file="m.yml",
        parameters_file="p.csv",
        data_files={"dataset_1": "d.nc"},
    )
    save_model(model, tmp_path / "m.yml")
    save_parameters(parameter, tmp_path / "p.csv")
    save_dataset(dataset, tmp_path / "d.nc")
    scheme_path = tmp_path / "testscheme.yml"
    save_scheme(file_name=scheme_path, format_name="yml", scheme=scheme)

    assert scheme_path.is_file()
    assert scheme_path.read_text() == want
    loaded = load_scheme(scheme_path)
    print(loaded.model.validate(loaded.parameters))
    assert loaded.model.valid(loaded.parameters)
    assert isinstance(scheme.data["dataset_1"], xr.Dataset)
