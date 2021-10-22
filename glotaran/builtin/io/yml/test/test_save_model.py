from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.examples.sequential import model
from glotaran.io import load_model
from glotaran.io import save_model

if TYPE_CHECKING:
    from pathlib import Path


want = """dataset:
  dataset1:
    group: default
    initial_concentration: j1
    irf: irf1
    megacomplex:
    - m1
dataset_groups:
  default:
    link_clp: null
    residual_function: variable_projection
default_megacomplex: decay
initial_concentration:
  j1:
    compartments:
    - s1
    - s2
    - s3
    exclude_from_normalize: []
    parameters:
    - j.1
    - j.0
    - j.0
irf:
  irf1:
    backsweep: false
    center: irf.center
    normalize: true
    type: gaussian
    width: irf.width
k_matrix:
  k1:
    matrix:
      (s2, s1): kinetic.1
      (s3, s2): kinetic.2
      (s3, s3): kinetic.3
megacomplex:
  m1:
    dimension: time
    k_matrix:
    - k1
    type: decay
"""


def test_save_model(
    tmp_path: Path,
):
    """Check all files exist."""

    model_path = tmp_path / "testmodel.yml"
    save_model(file_name=model_path, format_name="yml", model=model)

    assert model_path.is_file()
    assert model_path.read_text() == want
    assert load_model(model_path).valid()
