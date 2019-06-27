import pytest
from glotaran.parse.parser import load_yml_file
from glotaran.builtin.models.doas import (DOASModel, DOASMegacomplex, Oscillation)

# unused import
# from glotaran.model import FixedConstraint, BoundConstraint
# from os import getcwd
from os.path import join, dirname, abspath

THIS_DIR = dirname(abspath(__file__))


@pytest.fixture
def model():
    spec_path = join(THIS_DIR, 'test_model_spec_doas.yml')
    return load_yml_file(spec_path)


def test_correct_model(model):
    assert isinstance(model, DOASModel)


def test_oscillation(model):
    assert len(model.oscillation) == 2

    i = 1
    for _ in model.oscillation:
        label = "osc{}".format(i)
        assert label in model.oscillation
        oscillation = model.oscillation[label]
        assert isinstance(oscillation, Oscillation)
        assert oscillation.label == label
        assert oscillation.frequency == i
        assert oscillation.rate == 2+i

        i = i + 1


def test_megacomplexes(model):
    assert len(model.megacomplex) == 4

    i = 1
    for _ in model.megacomplex:
        label = "cmplx{}".format(i)
        assert label in model.megacomplex
        megacomplex = model.megacomplex[label]
        assert isinstance(megacomplex, DOASMegacomplex)
        assert megacomplex.label == label
        assert megacomplex.k_matrix == ["km{}".format(i)]
        if i == 2:
            assert megacomplex.oscillation == ["osc1"]
        if i == 4:
            assert megacomplex.oscillation == ["osc2"]

        i = i + 1
