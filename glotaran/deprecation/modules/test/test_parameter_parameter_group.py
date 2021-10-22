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
        label,value,minimum,maximum,vary,non-negative,expression
        j.1,1.0,-inf,inf,False,False,None
        j.0,0.0,-inf,inf,False,False,None
        kinetic.1,0.5,-inf,inf,True,False,None
        kinetic.2,0.3,-inf,inf,True,False,None
        kinetic.3,0.1,-inf,inf,True,False,None
        irf.center,0.3,-inf,inf,True,False,None
        irf.width,0.1,-inf,inf,True,False,None
        """
    )

    assert parameter_path.is_file()
    assert parameter_path.read_text() == expected
