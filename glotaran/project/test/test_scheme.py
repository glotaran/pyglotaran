from pathlib import Path

import pytest
import xarray as xr
from IPython.core.formatters import format_display_data

from glotaran.io import load_scheme
from glotaran.project import Scheme


@pytest.fixture
def mock_scheme(tmp_path: Path) -> Scheme:

    model_yml_str = """
    megacomplex:
        m1:
            type: decay
            k_matrix: []
    dataset:
        dataset1:
            megacomplex: [m1]
    """
    model_path = tmp_path / "model.yml"
    model_path.write_text(model_yml_str)

    parameter_path = tmp_path / "parameters.yml"
    parameter_path.write_text("[1.0, 67.0]")

    dataset_path = tmp_path / "dataset.nc"
    xr.DataArray([[1, 2, 3]], coords=[("spectral", [1]), ("time", [1, 2, 3])]).to_dataset(
        name="data"
    ).to_netcdf(dataset_path)

    scheme_yml_str = f"""
    model_file: {model_path}
    parameters_file: {parameter_path}
    maximum_number_function_evaluations: 42
    data_files:
        dataset1: {dataset_path}
    """
    scheme_path = tmp_path / "scheme.yml"
    scheme_path.write_text(scheme_yml_str)

    return load_scheme(scheme_path)


def test_scheme(mock_scheme: Scheme):
    """Test scheme attributes."""
    assert mock_scheme.model is not None

    assert mock_scheme.model_dimensions["dataset1"] == "time"
    assert mock_scheme.global_dimensions["dataset1"] == "spectral"
    assert mock_scheme.parameters is not None
    assert mock_scheme.parameters.get("1") == 1.0
    assert mock_scheme.parameters.get("2") == 67.0

    assert mock_scheme.maximum_number_function_evaluations == 42

    assert "dataset1" in mock_scheme.data
    assert mock_scheme.data["dataset1"].data.shape == (1, 3)


def test_scheme_ipython_rendering(mock_scheme: Scheme):
    """Autorendering in ipython"""

    rendered_obj = format_display_data(mock_scheme)[0]

    assert "text/markdown" in rendered_obj
    assert rendered_obj["text/markdown"].startswith("# Model")

    rendered_markdown_return = format_display_data(mock_scheme.markdown())[0]

    assert "text/markdown" in rendered_markdown_return
    assert rendered_markdown_return["text/markdown"].startswith("# Model")
