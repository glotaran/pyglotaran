"""Tests for glotaran.model.property.ModelProperty"""
from __future__ import annotations

from typing import Dict
from typing import List

from glotaran.model.property import ModelProperty
from glotaran.model.property import ParameterOrLabel
from glotaran.parameter import Parameter


def test_model_property_non_parameter():
    class MockClass:
        pass

    p_scalar = ModelProperty(MockClass, "scalar", int, "", None, True)
    assert p_scalar.glotaran_is_scalar_property
    assert not p_scalar.glotaran_is_sequence_property
    assert not p_scalar.glotaran_is_mapping_property
    assert p_scalar.glotaran_property_subtype is int
    assert not p_scalar.glotaran_is_parameter_property
    assert p_scalar.glotaran_value_as_markdown(42) == "42"

    p_sequence = ModelProperty(MockClass, "sequence", List[int], "", None, True)
    assert not p_sequence.glotaran_is_scalar_property
    assert p_sequence.glotaran_is_sequence_property
    assert not p_sequence.glotaran_is_mapping_property
    assert p_sequence.glotaran_property_subtype is int
    assert not p_sequence.glotaran_is_parameter_property
    print(p_sequence.glotaran_value_as_markdown([1, 2]))
    assert p_sequence.glotaran_value_as_markdown([1, 2]) == "\n  * 1\n  * 2"

    p_mapping = ModelProperty(MockClass, "mapping", Dict[str, int], "", None, True)
    assert not p_mapping.glotaran_is_scalar_property
    assert not p_mapping.glotaran_is_sequence_property
    assert p_mapping.glotaran_is_mapping_property
    assert p_mapping.glotaran_property_subtype is int
    assert not p_mapping.glotaran_is_parameter_property
    print(p_mapping.glotaran_value_as_markdown({"a": 1, "b": 2}))
    assert p_mapping.glotaran_value_as_markdown({"a": 1, "b": 2}) == "\n  * a: 1\n  * b: 2"


def test_model_property_parameter():
    class MockClass:
        pass

    p_scalar = ModelProperty(MockClass, "scalar", Parameter, "", None, True)
    assert p_scalar.glotaran_is_scalar_property
    assert not p_scalar.glotaran_is_sequence_property
    assert not p_scalar.glotaran_is_mapping_property
    assert p_scalar.glotaran_property_subtype is ParameterOrLabel
    assert p_scalar.glotaran_is_parameter_property

    p_sequence = ModelProperty(MockClass, "sequence", List[Parameter], "", None, True)
    assert not p_sequence.glotaran_is_scalar_property
    assert p_sequence.glotaran_is_sequence_property
    assert not p_sequence.glotaran_is_mapping_property
    assert p_sequence.glotaran_property_subtype is ParameterOrLabel
    assert p_sequence.glotaran_is_parameter_property

    p_mapping = ModelProperty(MockClass, "mapping", Dict[str, Parameter], "", None, True)
    assert not p_mapping.glotaran_is_scalar_property
    assert not p_mapping.glotaran_is_sequence_property
    assert p_mapping.glotaran_is_mapping_property
    assert p_mapping.glotaran_property_subtype is ParameterOrLabel
    assert p_mapping.glotaran_is_parameter_property


def test_model_property_default_getter():
    class MockClass:
        _p_default = None

    p_default = ModelProperty(MockClass, "p_default", int, "", 42, True)
    assert p_default.fget(MockClass) == 42
    MockClass._p_default = 21
    assert p_default.fget(MockClass) == 21


def test_model_property_parameter_setter():
    class MockClass:
        pass

    p_scalar = ModelProperty(MockClass, "scalar", Parameter, "", None, True)
    p_scalar.fset(MockClass, "param.foo")
    value = p_scalar.fget(MockClass)
    assert isinstance(value, Parameter)
    assert value.full_label == "param.foo"

    p_sequence = ModelProperty(MockClass, "sequence", List[Parameter], "", None, True)
    names = ["param1", "param2"]
    p_sequence.fset(MockClass, names)
    value = p_sequence.fget(MockClass)
    assert isinstance(value, list)
    assert all(isinstance(v, Parameter) for v in value)
    assert [p.full_label for p in value] == names

    p_mapping = ModelProperty(MockClass, "mapping", Dict[str, Parameter], "", None, True)
    p_mapping.fset(MockClass, {f"{i}": n for i, n in enumerate(names)})
    value = p_mapping.fget(MockClass)
    assert isinstance(value, dict)
    assert all(isinstance(v, Parameter) for v in value.values())
    assert [p.full_label for p in value.values()] == names


def test_model_property_parameter_to_label():
    class MockClass:
        pass

    p_scalar = ModelProperty(MockClass, "scalar", Parameter, "", None, True)
    p_scalar.fset(MockClass, "param.foo")
    value = p_scalar.fget(MockClass)
    assert p_scalar.glotaran_replace_parameter_with_labels(value) == "param.foo"

    p_sequence = ModelProperty(MockClass, "sequence", List[Parameter], "", None, True)
    names = ["param1", "param2"]
    p_sequence.fset(MockClass, names)
    value = p_sequence.fget(MockClass)
    assert p_sequence.glotaran_replace_parameter_with_labels(value) == names

    p_mapping = ModelProperty(MockClass, "mapping", Dict[str, Parameter], "", None, True)
    p_mapping.fset(MockClass, {f"{i}": n for i, n in enumerate(names)})
    value = p_mapping.fget(MockClass)
    assert list(p_mapping.glotaran_replace_parameter_with_labels(value).values()) == names
