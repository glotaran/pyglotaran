from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.examples.sequential import model
from glotaran.io import load_model
from glotaran.io import save_model

if TYPE_CHECKING:
    from py.path import local as TmpDir


want = """dataset:
  dataset_1:
    initial_concentration: initial_concentration_dataset_1
    irf: gaussian_irf
    megacomplex:
    - megacomplex_parallel_decay
default-megacomplex: decay
initial_concentration:
  initial_concentration_dataset_1:
    compartments:
    - species_1
    - species_2
    - species_3
    exclude_from_normalize: []
    parameters:
    - initial_concentration.1
    - initial_concentration.0
    - initial_concentration.0
irf:
  gaussian_irf:
    backsweep: false
    center: irf.center
    normalize: true
    type: gaussian
    width: irf.width
k_matrix:
  k_matrix_sequential:
    matrix:
      (species_2, species_1): decay.species_1
      (species_3, species_2): decay.species_2
      (species_3, species_3): decay.species_3
megacomplex:
  megacomplex_parallel_decay:
    dimension: time
    k_matrix:
    - k_matrix_sequential
    type: decay
"""


def test_save_model(
    tmpdir: TmpDir,
):
    """Check all files exist."""

    model_path = tmpdir / "testmodel.yml"
    save_model(file_name=model_path, format_name="yml", model=model)

    assert model_path.exists()
    with open(model_path) as f:
        got = f.read()
        assert got == want
    assert load_model(model_path).valid()
