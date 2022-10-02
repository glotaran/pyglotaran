"""Tests for ``glotaran.optimization.optimizer``."""

from textwrap import dedent

import pytest

from glotaran.optimization.optimizer import Optimizer


@pytest.mark.parametrize(
    "optimize_stdout, expected",
    (
        ("random string", 0),
        (
            dedent(
                """\
                   Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality
                """  # noqa: E501
            ),
            0,
        ),
        (
            dedent(
                """\
                   Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality
                       0              1         7.5834e+00                                    3.84e+01
                       1              2         7.5833e+00      1.37e-04       4.55e-05       1.26e-01
                """  # noqa: E501
            ),
            1,
        ),
        (
            dedent(
                """\
                   Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality
                       0              1         7.5834e+00                                    3.84e+01
                       1              2         7.5833e+00      1.37e-04       4.55e-05       1.26e-01
                       2              3         7.5833e+00      6.02e-11       6.44e-09       1.64e-05
                Both `ftol` and `xtol` termination conditions are satisfied.
                Function evaluations 3, initial cost 7.5834e+00, final cost 7.5833e+00, first-order optimality 1.64e-05.
                """  # noqa: E501
            ),
            2,
        ),
    ),
)
def test_optimizer_get_current_optimization_iteration(optimize_stdout: str, expected: int):
    """Test that the correct iteration is returned."""
    assert Optimizer.get_current_optimization_iteration(optimize_stdout) == expected
