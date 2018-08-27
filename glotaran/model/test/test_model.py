import pytest

from glotaran.fitmodel import FitModel, Matrix, Result
from glotaran.model import DatasetDescriptor, Model, glotaran_model, glotaran_model_item


@glotaran_model_item(attributes={
                        'p1': str,
                        'p2': str,
                     },
                     parameter=['p2'])
class MockAttr:
    pass


@glotaran_model('mock',
                attributes={"test": MockAttr},
                )
class MockModel(Model):
    pass


@pytest.fixture
def model():
    d = {
        "compartments": ['s1', 's2'],
        "megacomplex": {"m1": [], "m2": []},
        "initial_concentration": {
            "j1": [["1", "2"]],
            "j2": {'parameters': ["3", "4"]},
        },
        "test": {
            "t1": {'p1': "foo", 'p2': "baz"},
            "t2": ['hello', 'world'],
        }
    }
    return MockModel.from_dict(d)

def test_model_types(model):
    assert model.model_type == 'mock'
    assert model.dataset_type is DatasetDescriptor
    assert model.fitmodel_type is FitModel
    assert model.calculated_matrix is Matrix
    assert model.estimated_matrix is Matrix


@pytest.mark.parametrize("attr",
                         [
                             "megacomplex",
                             "initial_concentration",
                             "test"
])
def test_model_attr(model, attr):
    assert hasattr(model, attr)
    assert hasattr(model, f'get_{attr}')
    assert hasattr(model, f'set_{attr}')

def test_pars(model):
    assert model.compartments == ['s1', 's2']

    assert 'm1' in model.megacomplex
    assert 'm2' in model.megacomplex

    assert 'j1' in model.initial_concentration
    assert model.get_initial_concentration('j1').parameters == ['1', '2']
    assert 'j2' in model.initial_concentration
    assert model.get_initial_concentration('j2').parameters == ['3', '4']

    assert 't1' in model.test
    assert model.get_test('t1').p1 == 'foo'
    assert model.get_test('t1').p2 == 'baz'
    assert 't2' in model.test
    assert model.get_test('t2').p1 == 'hello'
    assert model.get_test('t2').p2 == 'world'
