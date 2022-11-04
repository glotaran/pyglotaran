from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.io import load_model
from glotaran.io import save_model
from glotaran.testing.simulated_data.sequential_spectral_decay import MODEL

if TYPE_CHECKING:
    from pathlib import Path


want = """\
clp_penalties: []
clp_constraints: []
clp_relations: []
dataset_groups:
  default:
    label: default
    residual_function: variable_projection
    link_clp: null
weights: []
megacomplex:
  megacomplex_sequential_decay:
    label: megacomplex_sequential_decay
    dimension: time
    compartments:
    - species_1
    - species_2
    - species_3
    rates:
    - rates.species_1
    - rates.species_2
    - rates.species_3
    type: decay-sequential
irf:
  gaussian_irf:
    label: gaussian_irf
    scale: null
    shift: null
    normalize: true
    backsweep: false
    backsweep_period: null
    type: gaussian
    center: irf.center
    width: irf.width
dataset:
  dataset_1:
    label: dataset_1
    group: default
    force_index_dependent: false
    megacomplex:
    - megacomplex_sequential_decay
    megacomplex_scale: null
    global_megacomplex: null
    global_megacomplex_scale: null
    scale: null
    irf: gaussian_irf
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
