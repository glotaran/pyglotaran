from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

import glotaran.builtin.io.yml.yml as yml_module
from glotaran.deprecation.modules.test import deprecation_warning_on_call_test_helper
from glotaran.io import load_model
from glotaran.io import load_scheme

if TYPE_CHECKING:
    from typing import Any

    from _pytest.monkeypatch import MonkeyPatch


@pytest.mark.parametrize(
    "model_yml_str, expected_nr_of_warnings, expected_key, expected_value",
    (
        ("type: kinetic-spectrum", 1, "default_megacomplex", "decay"),
        ("type: spectral-model", 1, "default_megacomplex", "spectral"),
        ("default-megacomplex: decay", 1, "default_megacomplex", "decay"),
        ("default-megacomplex: spectral", 1, "default_megacomplex", "spectral"),
        (
            dedent(
                """
                spectral_relations:
                    - compartment: s1
                    - compartment: s2
                """
            ),
            3,
            "clp_relations",
            [{"source": "s1"}, {"source": "s2"}],
        ),
        (
            dedent(
                """
                relations:
                    - compartment: s1
                    - compartment: s2
                """
            ),
            3,
            "clp_relations",
            [{"source": "s1"}, {"source": "s2"}],
        ),
        (
            dedent(
                """
                spectral_constraints:
                    - compartment: s1
                    - compartment: s2
                """
            ),
            3,
            "clp_constraints",
            [{"target": "s1"}, {"target": "s2"}],
        ),
        (
            dedent(
                """
                constraints:
                    - compartment: s1
                    - compartment: s2
                """
            ),
            3,
            "clp_constraints",
            [{"target": "s1"}, {"target": "s2"}],
        ),
        (
            dedent(
                """
                equal_area_penalties:
                    - type: equal_area
                """
            ),
            1,
            "clp_area_penalties",
            [{"type": "equal_area"}],
        ),
        (
            dedent(
                """
                irf:
                    irf1:
                        center_dispersion: [cdc1]

                """
            ),
            1,
            "irf",
            {"irf1": {"center_dispersion_coefficients": ["cdc1"]}},
        ),
        (
            dedent(
                """
                irf:
                    irf1:
                        "width_dispersion": [wdc1]

                """
            ),
            1,
            "irf",
            {"irf1": {"width_dispersion_coefficients": ["wdc1"]}},
        ),
    ),
    ids=(
        "type: kinetic-spectrum",
        "type: spectral-model",
        "default-megacomplex: decay",
        "default-megacomplex: spectral",
        "spectral_relations",
        "relations",
        "spectral_constraints",
        "constraints",
        "equal_area_penalties",
        "center_dispersion",
        "width_dispersion",
    ),
)
def test_model_spec_deprecations(
    monkeypatch: MonkeyPatch,
    model_yml_str: str,
    expected_nr_of_warnings: int,
    expected_key: str,
    expected_value: Any,
):
    """Warning gets emitted by load_model"""
    return_dicts = []
    with monkeypatch.context() as m:
        m.setattr(yml_module, "sanitize_yaml", lambda spec: return_dicts.append(spec))
        record, _ = deprecation_warning_on_call_test_helper(
            load_model, args=(model_yml_str,), kwargs={"format_name": "yml_str"}
        )

        return_dict = return_dicts[0]

        assert expected_key in return_dict
        assert return_dict[expected_key] == expected_value

        assert len(record) == expected_nr_of_warnings


@pytest.mark.parametrize(
    "scheme_yml_str, expected_key, expected_value",
    (
        ("maximum-number-function-evaluations: 12", "maximum_number_function_evaluations", 12),
        ("non-negative-least-squares: true", "non_negative_least_squares", True),
    ),
)
def test_scheme_spec_deprecations(
    monkeypatch: MonkeyPatch,
    scheme_yml_str: str,
    expected_key: str,
    expected_value: Any,
):
    """Warning gets emitted by load_model"""
    return_dicts = []
    with monkeypatch.context() as m:
        m.setattr(
            yml_module, "fromdict", lambda _, spec, *args, **kwargs: return_dicts.append(spec)
        )
        record, _ = deprecation_warning_on_call_test_helper(
            load_scheme, args=(scheme_yml_str,), kwargs={"format_name": "yml_str"}
        )

        return_dict = return_dicts[0]

        assert expected_key in return_dict
        assert return_dict[expected_key] == expected_value

        assert len(record) == 1
