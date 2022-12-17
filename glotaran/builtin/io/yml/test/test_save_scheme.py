from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr

from glotaran.io import load_scheme
from glotaran.io import save_dataset
from glotaran.io import save_model
from glotaran.io import save_parameters
from glotaran.io import save_scheme
from glotaran.project import Scheme
from glotaran.testing.simulated_data.sequential_spectral_decay import DATASET
from glotaran.testing.simulated_data.sequential_spectral_decay import MODEL
from glotaran.testing.simulated_data.sequential_spectral_decay import PARAMETERS
from glotaran.utils.io import chdir_context

want = """\
model: m.yml
parameters: p.csv
data:
  dataset_1: d.nc
clp_link_tolerance: 0.0
clp_link_method: nearest
maximum_number_function_evaluations: null
add_svd: true
ftol: 1e-08
gtol: 1e-08
xtol: 1e-08
optimization_method: TrustRegionReflection
result_path: null
"""


@pytest.mark.parametrize("path_is_absolute", (True, False))
def test_save_scheme(tmp_path: Path, path_is_absolute: bool):
    save_model(MODEL, tmp_path / "m.yml")
    save_parameters(PARAMETERS, tmp_path / "p.csv")
    save_dataset(DATASET, tmp_path / "d.nc")
    scheme = Scheme(
        MODEL,
        PARAMETERS,
        {"dataset_1": DATASET},
    )
    if path_is_absolute is True:
        scheme_path = tmp_path / "testscheme.yml"
    else:
        scheme_path = Path("testscheme.yml")

    with chdir_context("." if path_is_absolute is True else tmp_path):
        save_scheme(file_name=scheme_path, format_name="yml", scheme=scheme)

        assert scheme_path.is_file()
        assert scheme_path.read_text() == want
        loaded = load_scheme(scheme_path)
        print(loaded.model.validate(loaded.parameters))
        assert loaded.model.valid(loaded.parameters)
        assert isinstance(scheme.data["dataset_1"], xr.Dataset)
