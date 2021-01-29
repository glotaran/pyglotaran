from glotaran.model import Model
from glotaran.model import model
from glotaran.model import model_attribute
from glotaran.parse import register


def mock_matrix_fun():
    pass


@model_attribute()
class MockMegacomplex:
    pass


@model(
    "register_mock",
    attributes={},
    matrix=mock_matrix_fun,
    megacomplex_type=MockMegacomplex,
    model_dimension="c",
    global_dimension="e",
)
class MockModelRegister(Model):
    pass


def test_register():
    assert register.known_model("register_mock")
    assert issubclass(register.get_model("register_mock"), MockModelRegister)
