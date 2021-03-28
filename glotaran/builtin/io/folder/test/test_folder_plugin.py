from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from glotaran.analysis.optimize import optimize
from glotaran.analysis.simulation import simulate
from glotaran.analysis.test.models import ThreeDatasetDecay as suite
from glotaran.io import save_result
from glotaran.project import Scheme

if TYPE_CHECKING:
    from typing import Literal

    from py.path import local as TmpDir


@pytest.mark.parametrize("format_names", ("folder", "legacy"))
def test_save_result_folder(tmpdir: TmpDir, format_name: Literal["folder", "legacy"]):
    """Check all files exist."""
    model = suite.model

    model.is_grouped = False
    model.is_index_dependent = False

    wanted_parameters = suite.wanted_parameters
    data = {}
    for i in range(3):
        e_axis = getattr(suite, "e_axis" if i == 0 else f"e_axis{i+1}")
        c_axis = getattr(suite, "c_axis" if i == 0 else f"c_axis{i+1}")

        data[f"dataset{i+1}"] = simulate(
            suite.sim_model, f"dataset{i+1}", wanted_parameters, {"e": e_axis, "c": c_axis}
        )
    scheme = Scheme(
        model=suite.model,
        parameters=suite.initial_parameters,
        data=data,
        maximum_number_function_evaluations=1,
    )

    result = optimize(scheme)

    result_dir = Path(tmpdir / "testresult")
    save_result(result_path=str(result_dir), format_name=format_name, result=result)

    assert (result_dir / "result.md").exists()
    assert (result_dir / "optimized_parameters.csv").exists()
    assert (result_dir / "dataset1.nc").exists()
    assert (result_dir / "dataset2.nc").exists()
    assert (result_dir / "dataset3.nc").exists()
