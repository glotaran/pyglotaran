"""Test for ``glotaran.utils.tee``."""
from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.utils.tee import TeeContext

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture


def test_tee_context(capsys: CaptureFixture):
    """Test that print writes to stdout and is readable in context."""
    print_str = "foobar"
    expected = f"{print_str}\n"
    with TeeContext() as tee:
        print(print_str)
        result = tee.read()

    stdout, _ = capsys.readouterr()

    assert stdout == expected
    assert result == expected
