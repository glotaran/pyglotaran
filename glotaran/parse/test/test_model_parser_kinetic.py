import pytest

from glotaran.parse.parser import load_yml_file
from glotaran.models.spectral_temporal import (
    InitialConcentration,
    KineticModel,
    IrfGaussian,
    SpectralShapeGaussian,
    KineticMegacomplex,
    ZeroConstraint,
    EqualAreaConstraint,
)

from os.path import join, dirname, abspath
import numpy as np

THIS_DIR = dirname(abspath(__file__))


@pytest.fixture
def model():
    spec_path = join(THIS_DIR, 'test_model_spec_kinetic.yml')
    return load_yml_file(spec_path)


def test_correct_model(model):
    assert isinstance(model, KineticModel)


def test_dataset(model):
    assert len(model.dataset) == 2

    assert 'dataset1' in model.dataset
    dataset = model.dataset['dataset1']
    assert dataset.label == 'dataset1'
    assert dataset.megacomplex == ["cmplx1"]
    assert dataset.initial_concentration == "inputD1"
    assert dataset.irf == "irf1"
    assert dataset.scale == 1

    assert len(dataset.shapes) == 2
    assert dataset.shapes['s1'] == "shape1"
    assert dataset.shapes['s2'] == "shape2"

    dataset = model.dataset['dataset2']
    assert len(dataset.spectral_constraints) == 3

    assert any(isinstance(c, ZeroConstraint) for c in
               dataset.spectral_constraints)

    zcs = [zc for zc in dataset.spectral_constraints
           if zc.type == 'zero']
    assert len(zcs) == 2
    for zc in zcs:
        assert zc.compartment == 's1'
        assert zc.interval == [(1, 100), (2, 200)]

    assert any(isinstance(c, EqualAreaConstraint) for c in
               dataset.spectral_constraints)
    eac = [eac for eac in dataset.spectral_constraints
           if isinstance(eac, EqualAreaConstraint)][0]
    assert eac.compartment == 's3'
    assert eac.interval == [(670, 810)]
    assert eac.target == 's2'
    assert eac.parameter == 55
    assert eac.weight == 0.0016


def test_initial_concentration(model):
    assert len(model.initial_concentration) == 2

    i = 1
    for _ in model.initial_concentration:
        label = "inputD{}".format(i)
        assert label in model.initial_concentration
        initial_concentration = model.initial_concentration[label]
        assert initial_concentration.compartments == ['s1', 's2', 's3']
        assert isinstance(initial_concentration, InitialConcentration)
        assert initial_concentration.label == label
        assert initial_concentration.parameters == [1, 2, 3]


def test_irf(model):
    assert len(model.irf) == 2

    i = 1
    for _ in model.irf:
        label = "irf{}".format(i)
        assert label in model.irf
        irf = model.irf[label]
        assert isinstance(irf, IrfGaussian)
        assert irf.label == label
        want = [1] if i is 1 else [1, 2]
        assert irf.center == want
        want = [2] if i is 1 else [3, 4]
        assert irf.width == want
        want = [3] if i is 1 else [5, 6]
        assert irf.center_dispersion == want
        want = [4] if i is 1 else [7, 8]
        assert irf.width_dispersion == want
        want = None if i is 1 else 9
        assert irf.scale == want
        assert not irf.normalize

        if i is 2:
            assert irf.backsweep
            assert irf.backsweep_period, 55
        else:
            assert not irf.backsweep
            assert irf.backsweep_period is None

        i = i + 1


def test_k_matrices(model):
    assert "km1" in model.k_matrix
    print(model.k_matrix['km1'])
    assert np.array_equal(model.k_matrix["km1"].reduced(['s1', 's2', 's3', 's4']),
                          np.asarray([[1, 3, 5, 7],
                                      [2, 0, 0, 0],
                                      [4, 0, 0, 0],
                                      [6, 0, 0, 0]]))


def test_shapes(model):

    assert "shape1" in model.shape

    shape = model.shape["shape1"]
    assert isinstance(shape, SpectralShapeGaussian)
    assert shape.amplitude == "shape.1"
    assert shape.location == "shape.2"
    assert shape.width == "shape.3"


def test_megacomplexes(model):
    assert len(model.megacomplex) is 3

    i = 1
    for _ in model.megacomplex:
        label = "cmplx{}".format(i)
        assert label in model.megacomplex
        megacomplex = model.megacomplex[label]
        assert isinstance(megacomplex, KineticMegacomplex)
        assert megacomplex.label == label
        assert megacomplex.k_matrix == ["km{}".format(i)]
        i = i + 1
