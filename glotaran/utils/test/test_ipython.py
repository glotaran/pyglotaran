import pytest
from IPython.core.formatters import format_display_data

from glotaran.utils.ipython import MarkdownStr


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

    rendered_result = format_display_data(result)[0]

    assert "text/markdown" in rendered_result
    assert rendered_result["text/markdown"] == result_str
    assert rendered_result["text/plain"] == repr(raw_str)
