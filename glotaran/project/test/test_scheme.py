import pytest
import xarray as xr
from IPython.core.formatters import format_display_data

from glotaran.io import load_scheme
from glotaran.project import Scheme


@pytest.fixture
def mock_scheme(tmpdir):

    model_path = tmpdir.join("model.yml")
    with open(model_path, "w") as f:
        model = "type: kinetic-spectrum\ndataset:\n  dataset1:\n    megacomplex: []"
        f.write(model)

    parameter_path = tmpdir.join("parameters.yml")
    with open(parameter_path, "w") as f:
        parameter = "[1.0, 67.0]"
        f.write(parameter)

    dataset_path = tmpdir.join("dataset.nc")
    xr.DataArray([[1, 2, 3]], coords=[("e", [1]), ("c", [1, 2, 3])]).to_dataset(
        name="data"
    ).to_netcdf(dataset_path)

    scheme = f"""
    model: {model_path}
    parameters: {parameter_path}
    non-negative-least-squares: True
    maximum-number-function-evaluations: 42
    data:
      dataset1: {dataset_path}

    saving:
        level: minimal
        data_filter: [a, b, c]
        data_format: csv
        parameter_format: yaml
        report: false
    """
    scheme_path = tmpdir.join("scheme.yml")
    with open(scheme_path, "w") as f:
        f.write(scheme)

    yield load_scheme(scheme_path)


def test_scheme(mock_scheme: Scheme):
    assert mock_scheme.model is not None
    assert mock_scheme.model.model_type == "kinetic-spectrum"

    assert mock_scheme.parameters is not None
    assert mock_scheme.parameters.get("1") == 1.0
    assert mock_scheme.parameters.get("2") == 67.0

    assert mock_scheme.non_negative_least_squares
    assert mock_scheme.maximum_number_function_evaluations == 42

    assert "dataset1" in mock_scheme.data
    assert mock_scheme.data["dataset1"].data.shape == (1, 3)

    assert mock_scheme.saving.level == "minimal"
    assert mock_scheme.saving.data_filter == ["a", "b", "c"]
    assert mock_scheme.saving.data_format == "csv"
    assert mock_scheme.saving.parameter_format == "yaml"
    assert not mock_scheme.saving.report


def test_scheme_ipython_rendering(mock_scheme: Scheme):
    """Autorendering in ipython"""

    rendered_obj = format_display_data(mock_scheme)[0]

    assert "text/markdown" in rendered_obj
    assert rendered_obj["text/markdown"].startswith("# Model")

    rendered_markdown_return = format_display_data(mock_scheme.markdown())[0]

    assert "text/markdown" in rendered_markdown_return
    assert rendered_markdown_return["text/markdown"].startswith("# Model")
