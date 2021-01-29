from os.path import abspath
from os.path import dirname
from os.path import join

import numpy as np
import pytest

from glotaran.builtin.models.kinetic_image.initial_concentration import InitialConcentration
from glotaran.builtin.models.kinetic_image.irf import IrfMultiGaussian
from glotaran.builtin.models.kinetic_image.kinetic_image_megacomplex import KineticImageMegacomplex
from glotaran.builtin.models.kinetic_spectrum.kinetic_spectrum_dataset_descriptor import (
    KineticSpectrumDatasetDescriptor,
)
from glotaran.builtin.models.kinetic_spectrum.kinetic_spectrum_model import KineticSpectrumModel
from glotaran.builtin.models.kinetic_spectrum.spectral_constraints import ZeroConstraint
from glotaran.builtin.models.kinetic_spectrum.spectral_penalties import EqualAreaPenalty
from glotaran.builtin.models.kinetic_spectrum.spectral_shape import SpectralShapeGaussian
from glotaran.model import Weight
from glotaran.parameter import ParameterGroup
from glotaran.parse.parser import load_yaml_file

THIS_DIR = dirname(abspath(__file__))


@pytest.fixture
def model():
    spec_path = join(THIS_DIR, "test_model_spec_kinetic.yml")
    m = load_yaml_file(spec_path)
    print(m.markdown())
    return m


def test_correct_model(model):
    assert type(model).__name__ == "KineticSpectrumModel"
    assert isinstance(model, KineticSpectrumModel)


def test_dataset(model):
    assert len(model.dataset) == 2

    assert "dataset1" in model.dataset
    dataset = model.dataset["dataset1"]
    assert isinstance(dataset, KineticSpectrumDatasetDescriptor)
    assert dataset.label == "dataset1"
    assert dataset.megacomplex == ["cmplx1"]
    assert dataset.initial_concentration == "inputD1"
    assert dataset.irf == "irf1"
    assert dataset.scale == 1

    assert len(dataset.shape) == 2
    assert dataset.shape["s1"] == "shape1"
    assert dataset.shape["s2"] == "shape2"

    dataset = model.dataset["dataset2"]


def test_spectral_constraints(model):
    print(model.spectral_constraints)
    assert len(model.spectral_constraints) == 2

    assert any(isinstance(c, ZeroConstraint) for c in model.spectral_constraints)

    zcs = [zc for zc in model.spectral_constraints if zc.type == "zero"]
    assert len(zcs) == 2
    for zc in zcs:
        assert zc.compartment == "s1"
        assert zc.interval == [[1, 100], [2, 200]]


def test_spectral_penalties(model):
    assert len(model.equal_area_penalties) == 1
    assert all(isinstance(c, EqualAreaPenalty) for c in model.equal_area_penalties)
    eac = model.equal_area_penalties[0]
    assert eac.source == "s3"
    assert eac.source_intervals == [[670, 810]]
    assert eac.target == "s2"
    assert eac.target_intervals == [[670, 810]]
    assert eac.parameter == 55
    assert eac.weight == 0.0016


def test_spectral_relations(model):
    print(model.spectral_relations)
    assert len(model.spectral_relations) == 1

    rel = model.spectral_relations[0]

    assert rel.compartment == "s1"
    assert rel.target == "s2"
    assert rel.interval == [[1, 100], [2, 200]]


def test_initial_concentration(model):
    assert len(model.initial_concentration) == 2

    for i, _ in enumerate(model.initial_concentration, start=1):
        label = f"inputD{i}"
        assert label in model.initial_concentration
        initial_concentration = model.initial_concentration[label]
        assert initial_concentration.compartments == ["s1", "s2", "s3"]
        assert isinstance(initial_concentration, InitialConcentration)
        assert initial_concentration.label == label
        assert initial_concentration.parameters == [1, 2, 3]


def test_irf(model):
    assert len(model.irf) == 2

    for i, _ in enumerate(model.irf, start=1):
        label = f"irf{i}"
        assert label in model.irf
        irf = model.irf[label]
        assert isinstance(irf, IrfMultiGaussian)
        assert irf.label == label
        want = [1] if i == 1 else [1, 2]
        assert irf.center == want
        want = [2] if i == 1 else [3, 4]
        assert irf.width == want
        want = [3] if i == 1 else [5, 6]
        if i == 2:
            assert irf.center_dispersion == want
            want = [7, 8]
            assert irf.width_dispersion == want
            want = [9]
            assert irf.scale == want
        assert not irf.normalize

        if i == 2:
            assert irf.backsweep
            assert irf.backsweep_period, 55
        else:
            assert not irf.backsweep
            assert irf.backsweep_period is None


def test_k_matrices(model):
    assert "km1" in model.k_matrix
    parameter = ParameterGroup.from_list([1, 2, 3, 4, 5, 6, 7])
    reduced = model.k_matrix["km1"].fill(model, parameter).reduced(["s1", "s2", "s3", "s4"])
    assert np.array_equal(
        reduced, np.asarray([[1, 3, 5, 7], [2, 0, 0, 0], [4, 0, 0, 0], [6, 0, 0, 0]])
    )


def test_weight(model):

    weight = model.weights[0]
    assert isinstance(weight, Weight)
    assert weight.datasets == ["d1", "d2"]
    assert weight.global_interval == [100, 102]
    assert weight.model_interval == [301, 502]
    assert weight.value == 42


def test_shapes(model):

    assert "shape1" in model.shape

    shape = model.shape["shape1"]
    assert isinstance(shape, SpectralShapeGaussian)
    assert shape.amplitude.full_label == "shape.1"
    assert shape.location.full_label == "shape.2"
    assert shape.width.full_label == "shape.3"


def test_megacomplexes(model):
    assert len(model.megacomplex) == 3

    for i, _ in enumerate(model.megacomplex, start=1):
        label = f"cmplx{i}"
        assert label in model.megacomplex
        megacomplex = model.megacomplex[label]
        assert isinstance(megacomplex, KineticImageMegacomplex)
        assert megacomplex.label == label
        assert megacomplex.k_matrix == [f"km{i}"]
