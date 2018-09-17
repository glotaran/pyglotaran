import pytest
import numpy as np
from glotaran.model import ParameterGroup
from glotaran.models.spectral_temporal import InitialConcentration, KMatrix


class SequentialModel:
    params = [0.55, 0.0404, 1, 0]
    compartments = ['s1', 's2']
    matrix = {
        ('s2', 's1'): "1",
        ('s2', 's2'): "2",
    }
    jvec = ['3', '4']

    wanted_array = np.asarray([
        [0, 0],
        [0.55, 0.0404],
    ])

    wanted_full = np.asarray([
        [-0.55, 0],
        [0.55, -0.0404],
    ])

    wanted_eigen_vals = np.asarray([-0.55, -0.0404])

    wanted_eigen_vec = np.asarray([
        [0.6796527, 0],
        [-0.7335341, 1],
    ])

    wanted_gamma = [1.47134, 1.079278]

    wanted_a_matrix = np.asarray([
        [1, -1.079278],
        [0, 1.079278],
    ])


class SequentialModelWithBacktransfer:
    params = [0.55, 0.0404, 0.11, 1, 0]
    compartments = ['s1', 's2']
    matrix = {
        ('s2', 's1'): "1",
        ('s1', 's2'): "3",
        ('s2', 's2'): "2",
    }
    jvec = ['4', '5']

    wanted_array = np.asarray([
        [0, 0.11],
        [0.55, 0.0404],
    ])

    wanted_full = np.asarray([
        [-0.55, 0.11],
        [0.55, -0.1504],
    ])

    wanted_eigen_vals = np.asarray([-0.6670912, -0.03330879])

    wanted_eigen_vec = np.asarray([
        [-0.6846927, -0.2082266],
        [0.7288318, -0.9780806],
    ])

    wanted_gamma = [-1.19068, -0.8872538]

    wanted_a_matrix = np.asarray([
        [0.8152501, -0.8678057],
        [0.1847499, 0.8678057],
    ])


class ParallelModel:
    params = [0.55, 0.0404, 1]
    compartments = ['s1', 's2']
    matrix = {
        ('s1', 's1'): "1",
        ('s2', 's2'): "2",
    }
    jvec = ['3', '3']

    wanted_array = np.asarray([
        [0.55, 0],
        [0, 0.0404],
    ])

    wanted_full = np.asarray([
        [-0.55, 0],
        [0, -0.0404],
    ])

    wanted_eigen_vals = np.asarray([-0.55, -0.0404])

    wanted_eigen_vec = np.asarray([
        [1, 0],
        [0, 1],
    ])

    wanted_gamma = [1, 1]

    wanted_a_matrix = np.asarray([
        [1, 0],
        [0, 1],
    ])


class ParallelModelWithEquilibria:
    params = [0.55, 0.0404, 0.11, 0.02, 1]
    compartments = ['s1', 's2']
    matrix = {
        ('s1', 's1'): "4",
        ('s2', 's1'): "1",
        ('s2', 's2'): "2",
        ('s1', 's2'): "3",
    }
    jvec = ['5', '5']

    wanted_array = np.asarray([
        [0.02, 0.11],
        [0.55, 0.0404],
    ])

    wanted_full = np.asarray([
        [-0.57, 0.11],
        [0.55, -0.1504],
    ])

    wanted_eigen_vals = np.asarray([-0.6834894, -0.03691059])

    wanted_eigen_vec = np.asarray([
        [-0.6959817, -0.2020870],
        [0.7180595, -0.9793676],
    ])

    wanted_gamma = [-0.940184, -1.710398]

    wanted_a_matrix = np.asarray([
        [0.6543509, -0.6751081],
        [0.3456491, 1.6751081],
    ])


@pytest.mark.parametrize("matrix",
                         [
                             SequentialModel,
                             SequentialModelWithBacktransfer,
                             ParallelModel,
                             ParallelModelWithEquilibria
])
def test_matrix(matrix):

    params = ParameterGroup.from_list(matrix.params)

    mat = KMatrix("", matrix.matrix).fill(None, params)

    con = InitialConcentration("", matrix.jvec).fill(None, params)

    for comp in matrix.compartments:
        assert comp in mat.involved_compartments()

    print(mat.asarray(matrix.compartments))
    assert np.array_equal(mat.asarray(matrix.compartments), matrix.wanted_array)

    print(mat.full(matrix.compartments).T)
    assert np.allclose(mat.full(matrix.compartments), matrix.wanted_full)

    print(mat.eigen(matrix.compartments)[0])
    print(mat.eigen(matrix.compartments)[1])
    vals, vec = mat.eigen(matrix.compartments)
    assert np.allclose(vals, matrix.wanted_eigen_vals)
    assert np.allclose(vec, matrix.wanted_eigen_vec)

    print(mat._gamma(matrix.compartments, vec, con))
    assert np.allclose(mat._gamma(matrix.compartments, vec, con), matrix.wanted_gamma)

    print(mat.a_matrix(matrix.compartments, con))
    assert np.allclose(mat.a_matrix(matrix.compartments, con), matrix.wanted_a_matrix)
