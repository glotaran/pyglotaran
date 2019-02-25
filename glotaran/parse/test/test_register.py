from glotaran.model import Model, model, model_attribute

from glotaran.parse import register

#  from glotaran.analysis.test.mock import MockModel


@model_attribute()
class MockMegacomplex:
    pass


@model('register_mock',
       megacomplex_type=MockMegacomplex,
       )
class MockModel(Model):
    pass


def test_register():
    assert register.known_model('register_mock')
    assert issubclass(register.get_model('register_mock'), MockModel)
