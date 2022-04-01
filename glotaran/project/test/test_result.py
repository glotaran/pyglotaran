from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from IPython.core.formatters import format_display_data

from glotaran.analysis.optimize import optimize
from glotaran.io import SAVING_OPTIONS_DEFAULT
from glotaran.io import SAVING_OPTIONS_MINIMAL
from glotaran.io import SavingOptions
from glotaran.io import load_dataset
from glotaran.io import save_dataset
from glotaran.project.result import Result
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME
from glotaran.testing.simulated_data.shared_decay import SPECTRAL_AXIS


@pytest.fixture(scope="session")
def dummy_result():
    """Dummy result for testing."""
    print(SCHEME.data["dataset_1"])
    yield optimize(SCHEME, raise_exception=True)


def test_result_ipython_rendering(dummy_result: Result):
    """Autorendering in ipython"""

    rendered_obj = format_display_data(dummy_result)[0]

    assert "text/markdown" in rendered_obj
    assert rendered_obj["text/markdown"].startswith("| Optimization Result")

    rendered_markdown_return = format_display_data(dummy_result.markdown())[0]

    assert "text/markdown" in rendered_markdown_return
    assert rendered_markdown_return["text/markdown"].startswith("| Optimization Result")


def test_get_scheme(dummy_result: Result):
    scheme = dummy_result.get_scheme()
    assert "residual" not in dummy_result.scheme.data["dataset_1"]
    assert "residual" not in scheme.data["dataset_1"]
    assert all(scheme.parameters.to_dataframe() != dummy_result.scheme.parameters.to_dataframe())
    assert all(
        scheme.parameters.to_dataframe() == dummy_result.optimized_parameters.to_dataframe()
    )


def test_create_clp_guide_dataset(dummy_result: Result):
    """Check that clp guide has correct dimensions and dimension values."""
    clp_guide = dummy_result.create_clp_guide_dataset("dataset_1", "species_1")
    assert clp_guide.data.shape == (1, dummy_result.data["dataset_1"].spectral.size)
    assert np.allclose(clp_guide.coords["time"].item(), -1)
    assert np.allclose(clp_guide.coords["spectral"].values, SPECTRAL_AXIS)


def test_create_clp_guide_dataset_errors(dummy_result: Result):
    """Errors thrown when dataset or clp_label are not in result."""
    with pytest.raises(ValueError) as exc_info:
        dummy_result.create_clp_guide_dataset("not-a-dataset", "species_1")

    assert (
        str(exc_info.value)
        == "Unknown dataset 'not-a-dataset'. Known datasets are:\n ['dataset_1']"
    )
    with pytest.raises(ValueError) as exc_info:
        dummy_result.create_clp_guide_dataset("dataset_1", "not-a-species")

    assert (
        str(exc_info.value) == "Unknown clp_label 'not-a-species'. Known clp_labels are:\n "
        "['species_1', 'species_2', 'species_3']"
    )


def test_extract_sas_ascii_round_trip(dummy_result: Result, tmp_path: Path):
    """Save and load from ascii give same result."""
    tmp_file = tmp_path / "sas.ascii"

    sas = dummy_result.create_clp_guide_dataset("dataset_1", "species_1")
    with pytest.warns(UserWarning) as rec_warn:
        save_dataset(sas, tmp_file)

        assert len(rec_warn) == 1
        assert Path(rec_warn[0].filename).samefile(__file__)
        assert rec_warn[0].message.args[0] == (
            "Saving the 'data' attribute of 'dataset' as a fallback."
            "Result saving for ascii format only supports xarray.DataArray format, "
            "please pass a xarray.DataArray instead of a xarray.Dataset (e.g. dataset.data)."
        )

    loaded_sas = load_dataset(tmp_file, prepare=False)

    for dim in sas.dims:
        assert all(sas.coords[dim] == loaded_sas.coords[dim]), f"Coordinate {dim} mismatch"
    assert np.allclose(sas.data.values, loaded_sas.data.values)


@pytest.mark.parametrize("saving_options", [SAVING_OPTIONS_MINIMAL, SAVING_OPTIONS_DEFAULT])
def test_save_result(tmp_path: Path, saving_options: SavingOptions, dummy_result: Result):
    result_path = tmp_path / "test_result"
    dummy_result.save(result_path / "glotaran_result.yml", saving_options=saving_options)
    files_must_exist = [
        "glotaran_result.yml",
        "scheme.yml",
        "model.yml",
        "initial_parameters.csv",
        "optimized_parameters.csv",
        "parameter_history.csv",
        "dataset_1.nc",
    ]
    files_must_not_exist = []
    if saving_options.report:
        files_must_exist.append("result.md")
    else:
        files_must_not_exist.append("result.md")

    for file in files_must_exist:
        assert (result_path / file).exists()

    for file in files_must_not_exist:
        assert not (result_path / file).exists()

    dataset_path = result_path / "dataset_1.nc"
    assert dataset_path.exists()
    dataset = xr.open_dataset(dataset_path)
    print(dataset)
    if saving_options.data_filter is not None:
        assert len(saving_options.data_filter) == len(dataset)
        assert all(d in dataset for d in saving_options.data_filter)


def test_recreate(dummy_result):
    recreated_result = dummy_result.recreate()
    assert recreated_result.success


def test_verify(dummy_result):
    assert dummy_result.verify()
