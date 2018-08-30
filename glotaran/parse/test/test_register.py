from glotaran.model import Model, glotaran_model

from glotaran.parse import register

@glotaran_model('mock')
class MockModel(Model):
    pass


def test_register():
    assert register.known_model('mock')
    assert register.get_model('mock') is MockModel
