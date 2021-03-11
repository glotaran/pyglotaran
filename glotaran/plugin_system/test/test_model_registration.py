from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from glotaran.model import Model
from glotaran.plugin_system.base_registry import __PluginRegistry
from glotaran.plugin_system.model_registration import get_model
from glotaran.plugin_system.model_registration import known_model
from glotaran.plugin_system.model_registration import known_model_names
from glotaran.plugin_system.model_registration import register_model

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture
def mocked_registry(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(__PluginRegistry, "model", {"foo": Model, "bar": Model})


def test_register_model(mocked_registry):
    register_model("base-model", Model)

    assert "base-model" in __PluginRegistry.model
    assert __PluginRegistry.model["base-model"] == Model


def test_known_model(mocked_registry):
    assert known_model("foo")
    assert known_model("bar")
    assert not known_model("baz")


def test_get_model(mocked_registry):
    assert get_model("foo") == Model


def test_known_model_names(mocked_registry):
    assert known_model_names() == ["foo", "bar"]
