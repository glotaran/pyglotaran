from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from glotaran.io import load_parameters
from glotaran.io import save_parameters
from glotaran.parameter import ParameterGroup

PANDAS_TEST_DATA = Path(__file__).parent / "data"
PATH_XLSX = PANDAS_TEST_DATA / "reference_parameters.xlsx"
PATH_ODS = PANDAS_TEST_DATA / "reference_parameters.ods"
PATH_CSV = PANDAS_TEST_DATA / "reference_parameters.csv"
PATH_TSV = PANDAS_TEST_DATA / "reference_parameters.tsv"


@pytest.fixture(scope="module")
def yaml_reference() -> ParameterGroup:
    """Fixture for yaml reference data."""
    return load_parameters(PANDAS_TEST_DATA / "reference_parameters.yaml")


@pytest.mark.parametrize("reference_path", (PATH_XLSX, PATH_ODS, PATH_CSV, PATH_TSV))
def test_references(yaml_reference: ParameterGroup, reference_path: Path):
    """References are the same"""
    result = load_parameters(reference_path)
    assert result == yaml_reference


@pytest.mark.parametrize(
    "format_name,reference_path",
    (("xlsx", PATH_XLSX), ("ods", PATH_ODS), ("csv", PATH_CSV), ("tsv", PATH_TSV)),
)
def test_roundtrips(
    yaml_reference: ParameterGroup, tmp_path: Path, format_name: str, reference_path: Path
):
    """Roundtrip via save and load have the same data."""
    format_reference = load_parameters(reference_path)
    parameter_path = tmp_path / f"test_parameters.{format_name}"
    save_parameters(file_name=parameter_path, format_name=format_name, parameters=yaml_reference)
    parameters_roundtrip = load_parameters(parameter_path)

    assert parameters_roundtrip == yaml_reference
    assert parameters_roundtrip == format_reference

    if format_name in {"csv", "tsv"}:
        assert parameter_path.read_text() == reference_path.read_text()
    else:
        assert_frame_equal(
            pd.read_excel(parameter_path, na_values=["None", "none"]),
            pd.read_excel(reference_path, na_values=["None", "none"]),
        )
