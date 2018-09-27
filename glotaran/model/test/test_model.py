from typing import Dict, List, Tuple
import pytest

from glotaran.model import (
    BaseModel,
    DatasetDescriptor,
    ParameterGroup,
    model,
    model_item,
)


@model_item(
    attributes={
        'param': str,
        'megacomplex': str,
        'param_list': List[str],
        'default': {'type': int, 'default': 42},
        'complex': {'type': Dict[Tuple[str, str], str], 'target': ('compartment', 'parameter')},
    },
)
class MockAttr:
    pass


@model_item()
class MockMegacomplex:
    pass


@model('mock', attributes={"test": MockAttr}, megacomplex_type=MockMegacomplex)
class MockModel(BaseModel):
    pass


@pytest.fixture
def model():
    d = {
        "compartment": ['s1', 's2'],
        "megacomplex": {"m1": [], "m2": []},
        "initial_concentration": {
            "j1": [["1", "2"]],
            "j2": {'parameters': ["3", "4"]},
        },
        "test": {
            "t1": {'param': "foo",
                   'megacomplex': "m1",
                   'param_list': ["bar", "baz"],
                   'complex': {('s1', 's2'): "baz"},
                   },
            "t2": ['baz', 'm2', ['foo'], 7, {}],
        },
        "dataset": {
            "dataset1": {
                "megacomplex": ['m1', 'm2'],
                "scale": "scale_1",
                "compartment_constraints": [
                    {'type': 'zero',
                     'compartment': 's1',
                     'interval': [(0, 1)]},
                ]
            },
            "dataset2": [['m2'], 'scale_2', []]
        }
    }
    return MockModel.from_dict(d)


@pytest.fixture
def model_error():
    d = {
        "compartment": ['NOT_S1', 's2'],
        "megacomplex": {"m1": [], "m2": []},
        "test": {
            "t1": {'param': "fool",
                   'megacomplex': "mX",
                   'param_list': ["bar", "bay"],
                   'complex': {('s1', 's3'): "boz"},
                   },
        },
        "dataset": {
            "dataset1": {
                "megacomplex": ['N1', 'N2'],
                "scale": "scale_1",
                "compartment_constraints": [
                    {'type': 'zero',
                     'compartment': 's1',
                     'interval': [(0, 1)]},
                ]
            },
            "dataset2": [['mrX'], 'scale_3', None]
        }
    }
    return MockModel.from_dict(d)


@pytest.fixture
def parameter():
    params = [1, 2,
              ['foo', 3],
              ['bar', 4],
              ['baz', 2],
              ['scale_1', 2],
              ['scale_2', 8],
              4e2
              ]
    return ParameterGroup.from_list(params)


def test_model_types(model):
    assert model.model_type == 'mock'
    assert model.dataset_type is DatasetDescriptor


@pytest.mark.parametrize(
    "attr",
    ["dataset",
     "megacomplex",
     "test"])
def test_model_attr(model, attr):
    assert hasattr(model, attr)
    assert hasattr(model, f'get_{attr}')
    assert hasattr(model, f'set_{attr}')


def test_model_validity(model, model_error, parameter):
    print(model.errors())
    print(model.errors_parameter(parameter))
    assert model.valid()
    assert model.valid_parameter(parameter)
    print(model_error.errors())
    print(model_error.errors_parameter(parameter))
    assert not model_error.valid()
    assert len(model_error.errors()) is 7
    assert not model_error.valid_parameter(parameter)
    assert len(model_error.errors_parameter(parameter)) is 4


def test_items(model):
    assert model.compartment == ['s1', 's2']

    assert 'm1' in model.megacomplex
    assert 'm2' in model.megacomplex

    assert 't1' in model.test
    t = model.get_test('t1')
    assert t.param == "foo"
    assert t.megacomplex == 'm1'
    assert t.param_list == ["bar", "baz"]
    assert t.default == 42
    assert t.complex == {('s1', 's2'): "baz"}
    assert 't2' in model.test
    t = model.get_test('t2')
    assert t.param == "baz"
    assert t.megacomplex == 'm2'
    assert t.param_list == ["foo"]
    assert t.default == 7
    assert t.complex == {}

    assert 'dataset1' in model.dataset
    assert model.get_dataset('dataset1').megacomplex == ['m1', 'm2']
    assert model.get_dataset('dataset1').scale == 'scale_1'
    assert len(model.get_dataset('dataset1').compartment_constraints) == 1
    assert model.get_dataset('dataset1').compartment_constraints[0].type == 'zero'
    assert model.get_dataset('dataset1').compartment_constraints[0].interval == [(0, 1)]

    cons = model.get_dataset('dataset1').compartment_constraints[0]
    assert cons.type == 'zero'
    assert cons.compartment == 's1'
    assert cons.interval == [(0, 1)]

    assert 'dataset2' in model.dataset
    assert model.get_dataset('dataset2').megacomplex == ['m2']
    assert model.get_dataset('dataset2').scale == 'scale_2'
    assert len(model.get_dataset('dataset2').compartment_constraints) == 0


def test_fill(model, parameter):
    dataset = model.get_dataset('dataset1').fill(model, parameter)
    assert [cmplx.label for cmplx in dataset.megacomplex] == ['m1', 'm2']
    assert dataset.scale == 2
    assert len(dataset.compartment_constraints) == 1
    assert dataset.compartment_constraints[0].type == 'zero'
    assert dataset.compartment_constraints[0].interval == [(0, 1)]

    dataset = model.get_dataset('dataset2').fill(model, parameter)
    assert [cmplx.label for cmplx in dataset.megacomplex] == ['m2']
    assert dataset.scale == 8
    assert len(dataset.compartment_constraints) == 0

    t = model.get_test('t1').fill(model, parameter)
    assert t.param == 3
    assert t.megacomplex.label == 'm1'
    assert t.param_list == [4, 2]
    assert t.default == 42
    assert t.complex == {('s1', 's2'): 2}
    t = model.get_test('t2').fill(model, parameter)
    assert t.param == 2
    assert t.megacomplex.label == 'm2'
    assert t.param_list == [3]
    assert t.default == 7
    assert t.complex == {}
