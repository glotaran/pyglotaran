from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from glotaran.builtin.models.kinetic_image import KineticImageModel
from glotaran.model import Model
from glotaran.plugin_system.base_registry import __PluginRegistry
from glotaran.plugin_system.model_registration import get_model
from glotaran.plugin_system.model_registration import is_known_model
from glotaran.plugin_system.model_registration import known_model_names
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


def test_register_model(mocked_registry):
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


def test_is_known_model(mocked_registry):
    """Check if models are in registry"""
    assert is_known_model("foo")
    assert is_known_model("bar")
    assert not is_known_model("baz")


def test_get_model(mocked_registry):
    """Get model from registry"""
    assert get_model("foo") == Model


def test_known_model_names(mocked_registry):
    """Get model names from registry"""
    assert known_model_names() == sorted(["foo", "bar"])


def test_known_set_model_plugin(mocked_registry):
    """Overwrite foo model"""
    assert get_model("foo") == Model
    set_model_plugin("foo", "glotaran.builtin.models.kinetic_image.KineticImageModel")
    assert get_model("foo") == KineticImageModel


def test_known_set_model_plugin_dot_in_model_name(mocked_registry):
    """Raise error if model_name contains '.'"""
    with pytest.raises(
        ValueError,
        match=r"The value of 'model_name' isn't allowed to contain the character '\.' \.",
    ):
        set_model_plugin("foo.bar", "glotaran.builtin.models.kinetic_image.KineticImageModel")
