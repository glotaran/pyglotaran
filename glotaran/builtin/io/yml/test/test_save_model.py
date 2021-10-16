from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.examples.sequential_spectral_decay import MODEL
from glotaran.io import load_model
from glotaran.io import save_model

if TYPE_CHECKING:
    from pathlib import Path


want = """dataset:
  dataset_1:
    group: default
    irf: gaussian_irf
    megacomplex:
    - megacomplex_sequential_decay
dataset_groups:
  default:
    link_clp: null
    residual_function: variable_projection
default-megacomplex: decay-sequential
irf:
  gaussian_irf:
    backsweep: false
    center: irf.center
    normalize: true
    type: gaussian
    width: irf.width
megacomplex:
  megacomplex_sequential_decay:
    compartments:
    - species_1
    - species_2
    - species_3
    dimension: time
    rates:
    - rates.species_1
    - rates.species_2
    - rates.species_3
    type: decay-sequential
"""


def test_save_model(
    tmp_path: Path,
):
    """Check all files exist."""

    model_path = tmp_path / "testmodel.yml"
    save_model(file_name=model_path, format_name="yml", model=MODEL)

    assert model_path.is_file()
    assert model_path.read_text() == want
    assert load_model(model_path).valid()
