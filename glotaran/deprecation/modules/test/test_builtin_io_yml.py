from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

import glotaran.builtin.io.yml.yml as yml_module
from glotaran.deprecation.modules.test import deprecation_warning_on_call_test_helper
from glotaran.io import load_model

if TYPE_CHECKING:
    from typing import Any

    from _pytest.monkeypatch import MonkeyPatch


@pytest.mark.parametrize(
    "model_yml_str, expected_nr_of_warnings, expected_key, expected_value",
    (
        (
            dedent(
                """
                clp_area_penalties:
                    - type: equal_area
                """
            ),
            1,
            "clp_penalties",
            [{"type": "equal_area"}],
        ),
    ),
    ids=("clp_area_penalties",),
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
