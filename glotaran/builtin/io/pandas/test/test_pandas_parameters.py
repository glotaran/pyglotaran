from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from glotaran.io import load_parameters
from glotaran.io import save_parameters
from glotaran.parameter import Parameters

PANDAS_TEST_DATA = Path(__file__).parent / "data"
PATH_XLSX = PANDAS_TEST_DATA / "reference_parameters.xlsx"
PATH_ODS = PANDAS_TEST_DATA / "reference_parameters.ods"
PATH_CSV = PANDAS_TEST_DATA / "reference_parameters.csv"
PATH_TSV = PANDAS_TEST_DATA / "reference_parameters.tsv"
PATH_XLSX_ALT = PANDAS_TEST_DATA / "reference_parameters_alternative_notation.xlsx"
PATH_CSV_ALT = PANDAS_TEST_DATA / "reference_parameters_alternative_notation.csv"
PATH_CSV_SUBSET = PANDAS_TEST_DATA / "reference_parameters_subset.csv"


@pytest.fixture(scope="module")
def yaml_reference() -> Parameters:
    """Fixture for yaml reference data."""
    return load_parameters(PANDAS_TEST_DATA / "reference_parameters.yaml")


@pytest.fixture(scope="module")
def yaml_reference_subset() -> Parameters:
    """Fixture for yaml subset reference data."""
    return load_parameters(PANDAS_TEST_DATA / "reference_parameters_subset.yaml")


@pytest.mark.parametrize("reference_path", (PATH_XLSX, PATH_ODS, PATH_CSV, PATH_TSV))
def test_references(yaml_reference: Parameters, reference_path: Path):
    """References are the same"""
    result = load_parameters(reference_path)
    assert result == yaml_reference


def test_alternative_notations(yaml_reference: Parameters):
    """Reading parameter file with alternate syntax works and is case insensitive."""
    assert load_parameters(PATH_CSV_ALT) == yaml_reference
    assert load_parameters(PATH_XLSX_ALT) == yaml_reference


def test_csv_subset_notations(yaml_reference_subset: Parameters):
    """Reading parameter file with a susbset of possible attributes works."""
    assert load_parameters(PATH_CSV_SUBSET) == yaml_reference_subset


@pytest.mark.parametrize(
    "format_name,reference_path",
    (("xlsx", PATH_XLSX), ("ods", PATH_ODS), ("csv", PATH_CSV), ("tsv", PATH_TSV)),
)
def test_roundtrips(
    yaml_reference: Parameters, tmp_path: Path, format_name: str, reference_path: Path
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

        first_data_line = parameter_path.read_text().splitlines()[1]
        sep = "," if format_name == "csv" else "\t"

        assert f"{sep}-inf" not in first_data_line
        assert f"{sep}inf" not in first_data_line
    else:
        assert_frame_equal(
            pd.read_excel(parameter_path, na_values=["None", "none"]),
            pd.read_excel(reference_path, na_values=["None", "none"]),
        )


@pytest.mark.parametrize("format_name,sep", (("csv", ","), ("tsv", "\t")))
def test_replace_infinfinity(
    yaml_reference: Parameters, tmp_path: Path, format_name: str, sep: str
):
    parameter_path = tmp_path / f"test_parameters.{format_name}"
    save_parameters(
        file_name=parameter_path,
        format_name=format_name,
        parameters=yaml_reference,
        replace_infinfinity=False,
    )
    df = pd.read_csv(parameter_path, sep=sep)
    df = df[df["label"] != "verbose_list.no_defaults"]
    assert all(df["maximum"] == np.inf)
    assert all(df["minimum"] == -np.inf)

    first_data_line = parameter_path.read_text().splitlines()[1]
    assert f"{sep}-inf" in first_data_line
    assert f"{sep}inf" in first_data_line

    assert load_parameters(parameter_path) == yaml_reference
