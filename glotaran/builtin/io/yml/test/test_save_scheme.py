from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from pathlib import Path


want = """\
model: m.yml
parameters: p.csv
data:
  dataset_1: d.nc
clp_link_tolerance: 0.0
maximum_number_function_evaluations: null
add_svd: true
ftol: 1e-08
gtol: 1e-08
xtol: 1e-08
optimization_method: TrustRegionReflection
result_path: null
"""


def test_save_scheme(tmp_path: Path):
    save_model(MODEL, tmp_path / "m.yml")
    save_parameters(PARAMETERS, tmp_path / "p.csv")
    save_dataset(DATASET, tmp_path / "d.nc")
    scheme = Scheme(
        MODEL,
        PARAMETERS,
        {"dataset_1": DATASET},
    )
    scheme_path = tmp_path / "testscheme.yml"
    save_scheme(file_name=scheme_path, format_name="yml", scheme=scheme)

    assert scheme_path.is_file()
    assert scheme_path.read_text() == want
    loaded = load_scheme(scheme_path)
    print(loaded.model.validate(loaded.parameters))
    assert loaded.model.valid(loaded.parameters)
    assert isinstance(scheme.data["dataset_1"], xr.Dataset)
