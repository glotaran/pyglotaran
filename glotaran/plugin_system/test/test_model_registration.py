from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from glotaran.builtin.models.kinetic_image import KineticImageModel
from glotaran.model import Model
from glotaran.model import model
from glotaran.model.attribute import model_attribute
from glotaran.model.megacomplex import Megacomplex
from glotaran.plugin_system.base_registry import PluginOverwriteWarning
from glotaran.plugin_system.base_registry import __PluginRegistry
from glotaran.plugin_system.model_registration import get_model
from glotaran.plugin_system.model_registration import is_known_model
from glotaran.plugin_system.model_registration import known_model_names
from glotaran.plugin_system.model_registration import model_plugin_table
from glotaran.plugin_system.model_registration import register_model
from glotaran.plugin_system.model_registration import set_model_plugin

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture
def mocked_registry(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(
        __PluginRegistry,
        "model",
        {
            "foo": Model,
            "bar": KineticImageModel,
            "glotaran.builtin.models.kinetic_image.KineticImageModel": KineticImageModel,
        },
    )


@pytest.mark.usefixtures("mocked_registry")
def test_register_model():
    """Register new model."""
    register_model("base-model", Model)

    assert "base-model" in __PluginRegistry.model
    assert __PluginRegistry.model["base-model"] == Model
    assert "glotaran.model.base_model.Model" in __PluginRegistry.model
    assert __PluginRegistry.model["glotaran.model.base_model.Model"] == Model
    assert known_model_names(full_names=True) == sorted(
        [
            "foo",
            "bar",
            "glotaran.builtin.models.kinetic_image.KineticImageModel",
            "base-model",
            "glotaran.model.base_model.Model",
        ]
    )


@pytest.mark.usefixtures("mocked_registry")
def test_register_model_warning():
    """PluginOverwriteWarning raised pointing to correct file."""

    @model_attribute()
    class DummyAttr(Megacomplex):
        pass

    with pytest.warns(PluginOverwriteWarning, match="KineticImageModel.+bar.+Dummy") as record:

        @model(
            "bar",
            attributes={},
            megacomplex_types=DummyAttr,
            model_dimension="",
            global_dimension="",
        )
        class Dummy(Model):
            pass

        assert len(record) == 1
        assert Path(record[0].filename) == Path(__file__)


@pytest.mark.usefixtures("mocked_registry")
def test_is_known_model():
    """Check if models are in registry"""
    assert is_known_model("foo")
    assert is_known_model("bar")
    assert not is_known_model("baz")


@pytest.mark.usefixtures("mocked_registry")
def test_get_model():
    """Get model from registry"""
    assert get_model("foo") == Model


@pytest.mark.usefixtures("mocked_registry")
def test_known_model_names():
    """Get model names from registry"""
    assert known_model_names() == sorted(["foo", "bar"])


@pytest.mark.usefixtures("mocked_registry")
def test_known_set_model_plugin():
    """Overwrite foo model"""
    assert get_model("foo") == Model
    set_model_plugin("foo", "glotaran.builtin.models.kinetic_image.KineticImageModel")
    assert get_model("foo") == KineticImageModel


@pytest.mark.usefixtures("mocked_registry")
def test_known_set_model_plugin_dot_in_model_name():
    """Raise error if model_name contains '.'"""
    with pytest.raises(
        ValueError,
        match=r"The value of 'model_name' isn't allowed to contain the character '\.' \.",
    ):
        set_model_plugin("foo.bar", "glotaran.builtin.models.kinetic_image.KineticImageModel")


@pytest.mark.usefixtures("mocked_registry")
def test_model_plugin_table():
    """Short model table."""
    expected = dedent(
        """\
        |  __Model name__  |
        |------------------|
        |      `bar`       |
        |      `foo`       |
        """
    )

    assert f"{model_plugin_table()}\n" == expected


@pytest.mark.usefixtures("mocked_registry")
def test_model_plugin_table_full():
    """Full Table with all extras."""
    expected = dedent(
        """\
        |                      __Model name__                       |                                __Plugin name__                                |
        |-----------------------------------------------------------|-------------------------------------------------------------------------------|
        |                           `bar`                           | `glotaran.builtin.models.kinetic_image.kinetic_image_model.KineticImageModel` |
        |                           `foo`                           |                       `glotaran.model.base_model.Model`                       |
        | `glotaran.builtin.models.kinetic_image.KineticImageModel` | `glotaran.builtin.models.kinetic_image.kinetic_image_model.KineticImageModel` |
        """  # noqa: E501
    )

    assert f"{model_plugin_table(plugin_names=True,full_names=True)}\n" == expected
