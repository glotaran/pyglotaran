from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from glotaran.builtin.megacomplexes.decay import DecayMegacomplex
from glotaran.model import Megacomplex
from glotaran.model import megacomplex
from glotaran.plugin_system.base_registry import PluginOverwriteWarning
from glotaran.plugin_system.base_registry import __PluginRegistry
from glotaran.plugin_system.megacomplex_registration import get_megacomplex
from glotaran.plugin_system.megacomplex_registration import is_known_megacomplex
from glotaran.plugin_system.megacomplex_registration import known_megacomplex_names
from glotaran.plugin_system.megacomplex_registration import megacomplex_plugin_table
from glotaran.plugin_system.megacomplex_registration import register_megacomplex
from glotaran.plugin_system.megacomplex_registration import set_megacomplex_plugin

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture
def mocked_registry(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(
        __PluginRegistry,
        "megacomplex",
        {
            "foo": Megacomplex,
            "bar": DecayMegacomplex,
            "glotaran.builtin.megacomplexes.decay.DecayMegacomplex": DecayMegacomplex,
        },
    )


@pytest.mark.usefixtures("mocked_registry")
def test_register_megacomplex():
    """Register new megacomplex."""
    register_megacomplex("base-megacomplex", Megacomplex)

    assert "base-megacomplex" in __PluginRegistry.megacomplex
    assert __PluginRegistry.megacomplex["base-megacomplex"] == Megacomplex
    assert "glotaran.model.megacomplex.Megacomplex" in __PluginRegistry.megacomplex
    assert __PluginRegistry.megacomplex["glotaran.model.megacomplex.Megacomplex"] == Megacomplex
    assert known_megacomplex_names(full_names=True) == sorted(
        [
            "foo",
            "bar",
            "glotaran.builtin.megacomplexes.decay.DecayMegacomplex",
            "base-megacomplex",
            "glotaran.model.megacomplex.Megacomplex",
        ]
    )


@pytest.mark.usefixtures("mocked_registry")
def test_register_megacomplex_warning():
    """PluginOverwriteWarning raised pointing to correct file."""

    with pytest.warns(PluginOverwriteWarning, match="DecayMegacomplex.+bar.+Dummy") as record:

        @megacomplex(register_as="bar")
        class Dummy(DecayMegacomplex):
            pass

        assert len(record) == 1
        assert Path(record[0].filename) == Path(__file__)


@pytest.mark.usefixtures("mocked_registry")
def test_is_known_megacomplex():
    """Check if megacomplexs are in registry"""
    assert is_known_megacomplex("foo")
    assert is_known_megacomplex("bar")
    assert not is_known_megacomplex("baz")


@pytest.mark.usefixtures("mocked_registry")
def test_get_megacomplex():
    """Get megacomplex from registry"""
    assert get_megacomplex("foo") == Megacomplex


@pytest.mark.usefixtures("mocked_registry")
def test_known_megacomplex_names():
    """Get megacomplex names from registry"""
    assert known_megacomplex_names() == sorted(["foo", "bar"])


@pytest.mark.usefixtures("mocked_registry")
def test_known_set_megacomplex_plugin():
    """Overwrite foo megacomplex"""
    assert get_megacomplex("foo") == Megacomplex
    set_megacomplex_plugin("foo", "glotaran.builtin.megacomplexes.decay.DecayMegacomplex")
    assert get_megacomplex("foo") == DecayMegacomplex


@pytest.mark.usefixtures("mocked_registry")
def test_known_set_megacomplex_plugin_dot_in_megacomplex_name():
    """Raise error if megacomplex_name contains '.'"""
    with pytest.raises(
        ValueError,
        match=r"The value of 'megacomplex_name' isn't allowed to contain the character '\.' \.",
    ):
        set_megacomplex_plugin("foo.bar", "glotaran.builtin.megacomplexes.decay.DecayMegacomplex")


@pytest.mark.usefixtures("mocked_registry")
def test_megacomplex_plugin_table():
    """Short megacomplex table."""
    expected = dedent(
        """\
        |  __Megacomplex name__  |
        |------------------------|
        |         `bar`          |
        |         `foo`          |
        """
    )
    print(f"{megacomplex_plugin_table()}\n")
    assert f"{megacomplex_plugin_table()}\n" == expected


@pytest.mark.usefixtures("mocked_registry")
def test_megacomplex_plugin_table_full():
    """Full Table with all extras."""
    expected = dedent(
        """\
        |                  __Megacomplex name__                   |                              __Plugin name__                              |
        |---------------------------------------------------------|---------------------------------------------------------------------------|
        |                          `bar`                          | `glotaran.builtin.megacomplexes.decay.decay_megacomplex.DecayMegacomplex` |
        |                          `foo`                          |                 `glotaran.model.megacomplex.Megacomplex`                  |
        | `glotaran.builtin.megacomplexes.decay.DecayMegacomplex` | `glotaran.builtin.megacomplexes.decay.decay_megacomplex.DecayMegacomplex` |
        """  # noqa: E501
    )

    assert f"{megacomplex_plugin_table(plugin_names=True,full_names=True)}\n" == expected
