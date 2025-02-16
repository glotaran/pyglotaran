from __future__ import annotations

import numpy as np
import pytest

from glotaran.builtin.elements.kinetic import Kinetic


class SequentialModel:
    params = [0.55, 0.0404]
    species = ["s1", "s2"]
    rates = {
        ("s2", "s1"): 0.55,
        ("s2", "s2"): 0.0404,
    }
    initial_concentration = [1, 0]

    wanted_array = np.asarray(
        [
            [0, 0],
            [0.55, 0.0404],
        ]
    )

    wanted_full = np.asarray(
        [
            [-0.55, 0],
            [0.55, -0.0404],
        ]
    )

    wanted_eigen_vals = np.asarray([-0.55, -0.0404])

    wanted_eigen_vec = np.asarray(
        [
            [0.6796527, 0],
            [-0.7335341, 1],
        ]
    )

    wanted_a_matrix = np.asarray(
        [
            [1, -1.079278],
            [0, 1.079278],
        ]
    )


class SequentialModelWithBacktransfer:
    params = [0.55, 0.0404, 0.11]
    species = ["s1", "s2"]
    rates = {
        ("s2", "s1"): 0.55,
        ("s2", "s2"): 0.0404,
        ("s1", "s2"): 0.11,
    }
    initial_concentration = [1, 0]

    wanted_array = np.asarray(
        [
            [0, 0.11],
            [0.55, 0.0404],
        ]
    )

    wanted_full = np.asarray(
        [
            [-0.55, 0.11],
            [0.55, -0.1504],
        ]
    )

    wanted_eigen_vals = np.asarray([-0.6670912, -0.03330879])

    wanted_eigen_vec = np.asarray(
        [
            [-0.6846927, -0.2082266],
            [0.7288318, -0.9780806],
        ]
    )

    wanted_a_matrix = np.asarray(
        [
            [0.8152501, -0.8678057],
            [0.1847499, 0.8678057],
        ]
    )


class ParallelModel:
    params = [0.55, 0.0404]
    species = ["s1", "s2"]
    rates = {
        ("s1", "s1"): 0.55,
        ("s2", "s2"): 0.0404,
    }
    initial_concentration = [1, 1]

    wanted_array = np.asarray(
        [
            [0.55, 0],
            [0, 0.0404],
        ]
    )

    wanted_full = np.asarray(
        [
            [-0.55, 0],
            [0, -0.0404],
        ]
    )

    wanted_eigen_vals = np.asarray([-0.55, -0.0404])

    wanted_eigen_vec = np.asarray(
        [
            [1, 0],
            [0, 1],
        ]
    )

    wanted_a_matrix = np.asarray(
        [
            [1, 0],
            [0, 1],
        ]
    )


class ParallelModelWithEquilibria:
    params = [0.55, 0.0404, 0.11, 0.02]
    species = ["s1", "s2"]
    rates = {
        ("s2", "s1"): 0.55,
        ("s2", "s2"): 0.0404,
        ("s1", "s2"): 0.11,
        ("s1", "s1"): 0.02,
    }
    initial_concentration = [1, 1]

    wanted_array = np.asarray(
        [
            [0.02, 0.11],
            [0.55, 0.0404],
        ]
    )

    wanted_full = np.asarray(
        [
            [-0.57, 0.11],
            [0.55, -0.1504],
        ]
    )

    wanted_eigen_vals = np.asarray([-0.6834894, -0.03691059])

    wanted_eigen_vec = np.asarray(
        [
            [-0.6959817, -0.2020870],
            [0.7180595, -0.9793676],
        ]
    )

    wanted_a_matrix = np.asarray(
        [
            [0.6543509, -0.6751081],
            [0.3456491, 1.6751081],
        ]
    )


@pytest.mark.parametrize(
    "model",
    [SequentialModel, SequentialModelWithBacktransfer, ParallelModel, ParallelModelWithEquilibria],
)
def test_a_matrix_general(model):  # noqa: ANN001
    kinetic = Kinetic(rates=model.rates)
    assert kinetic.compartments == model.species

    initial_concentration = model.initial_concentration

    print(kinetic.array)
    print(model.wanted_array)
    assert np.array_equal(kinetic.array, model.wanted_array)

    print(kinetic.full_array)
    print(model.wanted_full)
    assert np.allclose(kinetic.full_array, model.wanted_full)

    print(kinetic.eigen()[0])
    print(kinetic.eigen()[1])
    vals, vec = kinetic.eigen()
    assert np.allclose(vals, model.wanted_eigen_vals)
    assert np.allclose(vec, model.wanted_eigen_vec)

    print(kinetic.a_matrix_general(initial_concentration))
    assert np.allclose(
        kinetic.a_matrix_general(initial_concentration),
        model.wanted_a_matrix,
    )


def test_a_matrix_sequential():
    rates = {
        ("s2", "s1"): 1,
        ("s3", "s2"): 4,
        ("s2", "s2"): 4,
        ("s3", "s3"): 5,
    }

    kinetic = Kinetic(rates=rates)

    initial_concentration = [1, 0, 0]

    assert not kinetic.is_sequential(initial_concentration)

    rates = {
        ("s2", "s1"): 0.55,
        ("s2", "s2"): 0.0404,
    }

    kinetic = Kinetic(rates=rates)

    initial_concentration = [1, 0]

    assert kinetic.is_sequential(initial_concentration)

    wanted_a_matrix = np.asarray(
        [
            [1, -1.079278],
            [0, 1.079278],
        ]
    )

    print(kinetic.a_matrix_sequential())
    assert np.allclose(kinetic.a_matrix_sequential(), wanted_a_matrix)


def test_combine_matrices():
    matrix1 = {
        ("s1", "s1"): 1,
        ("s2", "s2"): 2,
    }
    mat1 = Kinetic(rates=matrix1)
    matrix2 = {
        ("s2", "s2"): 3,
        ("s3", "s3"): 4,
    }
    mat2 = Kinetic(rates=matrix2)

    combined = Kinetic.combine([mat1, mat2])

    assert combined.rates[("s1", "s1")] == 1
    assert combined.rates[("s2", "s2")] == 3
    assert combined.rates[("s3", "s3")] == 4
