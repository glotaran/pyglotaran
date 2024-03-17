"""Tests for ``glotaran.utils.json_schema``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from glotaran.parameter import Parameters
from glotaran.utils.json_schema import create_model_scheme_json_schema

if TYPE_CHECKING:
    from pathlib import Path


def test_create_model_scheme_json_schema(tmp_path: Path):
    """Schema are created reproducible and parameters are strings."""
    blank_schema = create_model_scheme_json_schema()

    assert "properties" not in blank_schema["$defs"]["Parameter"]
    assert "activation" in blank_schema["$defs"]["GlotaranDataModel"]["properties"]
    assert create_model_scheme_json_schema() == create_model_scheme_json_schema()

    params = Parameters.from_dict({"foo": {"bar": [1]}})
    schema_with_params = create_model_scheme_json_schema(parameters=params)
    assert schema_with_params["$defs"]["Parameter"]["enum"] == ["foo.bar.1"]

    test_file = tmp_path / "schema.json"
    create_model_scheme_json_schema(test_file)
    assert json.loads(test_file.read_text()) == blank_schema

    test_file2 = tmp_path / "schema.json"
    create_model_scheme_json_schema(test_file2.as_posix())
    assert json.loads(test_file2.read_text()) == blank_schema
