from __future__ import annotations

import warnings
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

import glotaran.builtin.io.yml.yml as yml_module
from glotaran.io import load_model

if TYPE_CHECKING:
    from typing import Any

    from _pytest.monkeypatch import MonkeyPatch


@pytest.mark.parametrize(
    "model_yml_str, expected_nr_of_warnings, expected_key, expected_value",
    (
        ("type: kinetic-spectrum", 1, "default-megacomplex", "decay"),
        ("type: spectrum", 1, "default-megacomplex", "spectral"),
        (
            dedent(
                """
                spectral_relations:
                    - compartment: s1
                    - compartment: s2
                """
            ),
            3,
            "relations",
            [{"source": "s1"}, {"source": "s2"}],
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
    ),
    ids=("type: kinetic-spectrum", "type: spectrum", "spectral_relations", "equal_area_penalties"),
)
def test_model_spec_deprecations(
    monkeypatch: MonkeyPatch,
    model_yml_str: str,
    expected_nr_of_warnings: int,
    expected_key: str,
    expected_value: Any,
):
    """Warning gets emitted by load_model"""
    warnings.simplefilter("always", DeprecationWarning)
    return_dicts = []
    with monkeypatch.context() as m:
        m.setattr(yml_module, "sanitize_yaml", lambda spec: return_dicts.append(spec))
        with pytest.warns(DeprecationWarning) as record:
            try:
                load_model(model_yml_str, format_name="yml_str")
            except Exception:
                pass

            return_dict = return_dicts[0]

            assert expected_key in return_dict
            assert return_dict[expected_key] == expected_value

            assert len(record) == expected_nr_of_warnings
            assert Path(record[0].filename) == Path(__file__)
