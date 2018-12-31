from glotaran.model import BaseModel, model, model_item

from glotaran.parse import register

#  from glotaran.analysis.test.mock import MockModel


@model_item()
class MockMegacomplex:
    pass


@model('register_mock',
       megacomplex_type=MockMegacomplex,
       )
class MockModel(BaseModel):
    pass


def test_register():
    assert register.known_model('register_mock')
    assert issubclass(register.get_model('register_mock'), MockModel)
