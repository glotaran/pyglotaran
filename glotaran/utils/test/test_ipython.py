from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from IPython.core.formatters import format_display_data

from glotaran.utils.ipython import MarkdownStr
from glotaran.utils.ipython import display_file

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "raw_str, result_str, syntax",
    (
        ("# Model", "# Model", None),
        ("kinetic:\n  - ['1', 1]", "```yaml\nkinetic:\n  - ['1', 1]\n```", "yaml"),
    ),
)
def test_markdown_str_render(raw_str: str, result_str: str, syntax: str):
    """Rendering"""
    result = MarkdownStr(raw_str, syntax=syntax)

    assert str(result) == result_str
    assert result == result_str

    rendered_result = format_display_data(result)[0]

    assert "text/markdown" in rendered_result
    assert rendered_result["text/markdown"] == result_str
    assert rendered_result["text/plain"] == repr(raw_str)


def test_display_file(tmp_path: Path):
    """str and PathLike give the same result"""
    file_content = "kinetic:\n  - ['1', 1]"
    expected = MarkdownStr(file_content, syntax="yaml")
    tmp_file = tmp_path / "test.yml"
    tmp_file.write_text(file_content)

    assert display_file(tmp_file, syntax="yaml") == expected
    assert display_file(str(tmp_file), syntax="yaml") == expected
