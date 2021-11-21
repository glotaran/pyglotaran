"""Tests for deprecated methods in ``glotaran..parameter.ParameterGroup``."""
from pathlib import Path
from textwrap import dedent

from glotaran.deprecation.modules.test import deprecation_warning_on_call_test_helper
from glotaran.examples.sequential import parameter


def test_parameter_group_to_csv(tmp_path: Path):
    """``ParameterGroup.to_csv`` raises deprecation warning and saves file."""
    parameter_path = tmp_path / "test_parameter.csv"
    deprecation_warning_on_call_test_helper(
        parameter.to_csv, args=[parameter_path.as_posix()], raise_exception=True
    )
    expected = dedent(
        """\
        label,value,expression,minimum,maximum,non-negative,vary,standard-error
        j.1,1.0,None,-inf,inf,False,False,0.0
        j.0,0.0,None,-inf,inf,False,False,0.0
        kinetic.1,0.5,None,-inf,inf,False,True,0.0
        kinetic.2,0.3,None,-inf,inf,False,True,0.0
        kinetic.3,0.1,None,-inf,inf,False,True,0.0
        irf.center,0.3,None,-inf,inf,False,True,0.0
        irf.width,0.1,None,-inf,inf,False,True,0.0
        """
    )

    assert parameter_path.is_file()
    assert parameter_path.read_text() == expected
