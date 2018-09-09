import pytest
from glotaran.parse.parser import load_yml_file
from glotaran.models.damped_oscillation import (DOASModel,
                                                DOASMegacomplex,
                                                Oscillation)

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
    assert len(model.oscillations) == 2

    i = 1
    for _ in model.oscillations:
        label = "osc{}".format(i)
        assert label in model.oscillations
        oscillation = model.oscillations[label]
        assert isinstance(oscillation, Oscillation)
        assert oscillation.label == label
        assert oscillation.compartment == f"os{i}"
        assert oscillation.frequency == i
        assert oscillation.rate == 2+i

        i = i + 1


def test_megacomplexes(model):
    assert len(model.megacomplexes) == 4

    i = 1
    for _ in model.megacomplexes:
        label = "cmplx{}".format(i)
        assert label in model.megacomplexes
        megacomplex = model.megacomplexes[label]
        assert isinstance(megacomplex, DOASMegacomplex)
        assert megacomplex.label == label
        assert megacomplex.k_matrices == ["km{}".format(i)]
        if i is 2:
            assert megacomplex.oscillations == ["osc1"]
        if i is 4:
            assert megacomplex.oscillations == ["osc2"]

        i = i + 1
