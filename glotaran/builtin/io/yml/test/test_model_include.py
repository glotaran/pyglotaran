from pathlib import Path

from glotaran.io import load_model

TEST_MODEL_DATASET = """\
dataset:
  d1:
    megacomplex: [m1]
"""
TEST_MODEL_MEGACOMPLEX = """\
megacomplex:
  m1:
    type: decay
    k_matrix: []
"""

TEST_MODEL = """\
include:
  - {d_path}
  - {m_path}
"""


def test_model_include(tmp_path: Path):
    d_path = tmp_path / "dataset.yml"
    m_path = tmp_path / "megacomplex.yml"
    model_path = tmp_path / "model.yml"

    with open(d_path, "w") as f:
        f.write(TEST_MODEL_DATASET)

    with open(m_path, "w") as f:
        f.write(TEST_MODEL_MEGACOMPLEX)
    with open(model_path, "w") as f:
        f.write(TEST_MODEL.format(d_path=d_path, m_path=m_path))

    model = load_model(model_path)

    assert "d1" in model.dataset
    assert "m1" in model.megacomplex
