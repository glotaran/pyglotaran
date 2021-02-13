from glotaran import model
from glotaran.model.test.test_model import MockModel


def test_register():
    assert model.known_model("mock_model")
    assert issubclass(model.get_model("mock_model"), MockModel)
