from glotaran.model import BaseModel, model, model_item

from glotaran.parse import register

#  from glotaran.analysis.test.mock import MockModel


@model_item()
class MockMegacomplex:
    pass

@model('regeister_mock',
       megacomplex_type=MockMegacomplex,
       )
class MockModel(BaseModel):
    pass


def test_register():
    assert register.known_model('regeister_mock')
    assert issubclass(register.get_model('regeister_mock'), MockModel)
