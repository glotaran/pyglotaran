from glotaran.model import BaseModel, model

from glotaran.parse import register

from glotaran.analysis.test.mock import MockModel


def test_register():
    assert register.known_model('mock')
    assert issubclass(register.get_model('mock'), MockModel)
