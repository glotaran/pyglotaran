from glotaran.model import Model, model, model_attribute

from glotaran.parse import register


def mock_matrix_fun():
    pass


@model_attribute()
class MockMegacomplex:
    pass


@model('register_mock',
       matrix=mock_matrix_fun,
       megacomplex_type=MockMegacomplex,
       )
class MockModel(Model):
    pass


def test_register():
    assert register.known_model('register_mock')
    assert issubclass(register.get_model('register_mock'), MockModel)
