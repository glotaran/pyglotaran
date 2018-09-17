from glotaran.model import BaseModel, model

from glotaran.parse import register


@model('mock')
class MockModel(BaseModel):
    pass


def test_register():
    assert register.known_model('mock')
    assert register.get_model('mock') is MockModel
