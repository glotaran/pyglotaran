from __future__ import annotations

import numpy as np

from glotaran.io import load_parameters


def test_parameter_group_copy():
    params = """
    a:
        - ["foo", 1, {non-negative: true, min: -1, max: 1, vary: false}]
        - 4
        - 5
    b:
        - 7
        - 8
    """
    parameters = load_parameters(params, format_name="yml_str")

    assert parameters.get("a.foo").value == 1
    assert parameters.get("a.foo").non_negative
    assert parameters.get("a.foo").minimum == -1
    assert parameters.get("a.foo").maximum == 1
    assert not parameters.get("a.foo").vary

    assert parameters.get("a.2").value == 4
    assert not parameters.get("a.2").non_negative
    assert parameters.get("a.2").minimum == -np.inf
    assert parameters.get("a.2").maximum == np.inf
    assert parameters.get("a.2").vary

    assert parameters.get("a.3").value == 5

    assert parameters.get("b.1").value == 7
    assert not parameters.get("b.1").non_negative
    assert parameters.get("b.1").minimum == -np.inf
    assert parameters.get("b.1").maximum == np.inf
    assert parameters.get("b.1").vary

    assert parameters.get("b.2").value == 8
