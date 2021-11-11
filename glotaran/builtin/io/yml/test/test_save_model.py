from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.examples.sequential import model
from glotaran.io import load_model
from glotaran.io import save_model

if TYPE_CHECKING:
    from pathlib import Path


want = """\
default_megacomplex: decay
dataset_groups:
  default:
    residual_function: variable_projection
    link_clp: null
k_matrix:
  k1:
    matrix:
      (s2, s1): kinetic.1
      (s3, s2): kinetic.2
      (s3, s3): kinetic.3
initial_concentration:
  j1:
    compartments:
      - s1
      - s2
      - s3
    parameters:
      - j.1
      - j.0
      - j.0
    exclude_from_normalize: []
irf:
  irf1:
    type: gaussian
    center: irf.center
    width: irf.width
    normalize: true
    backsweep: false
megacomplex:
  m1:
    type: decay
    dimension: time
    k_matrix:
      - k1
dataset:
  dataset1:
    group: default
    megacomplex:
      - m1
    initial_concentration: j1
    irf: irf1
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
