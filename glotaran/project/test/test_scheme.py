import xarray as xr

from glotaran.io import load_scheme


def test_scheme(tmpdir):

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

    scheme = load_scheme(scheme_path)
    assert scheme.model is not None
    assert scheme.model.model_type == "kinetic-spectrum"

    assert scheme.parameters is not None
    assert scheme.parameters.get("1") == 1.0
    assert scheme.parameters.get("2") == 67.0

    assert scheme.non_negative_least_squares
    assert scheme.maximum_number_function_evaluations == 42

    assert "dataset1" in scheme.data
    assert scheme.data["dataset1"].data.shape == (1, 3)

    assert scheme.saving.level == "minimal"
    assert scheme.saving.data_filter == ["a", "b", "c"]
    assert scheme.saving.data_format == "csv"
    assert scheme.saving.parameter_format == "yaml"
    assert not scheme.saving.report
