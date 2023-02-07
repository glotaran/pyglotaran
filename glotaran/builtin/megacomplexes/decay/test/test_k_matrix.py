import numpy as np
import pytest
from IPython.core.formatters import format_display_data

from glotaran.builtin.megacomplexes.decay.k_matrix import KMatrix
from glotaran.builtin.megacomplexes.decay.k_matrix import calculate_gamma
from glotaran.model.item import fill_item
from glotaran.parameter import Parameters


class SequentialModel:
    params = [0.55, 0.0404]
    compartments = ["s1", "s2"]
    matrix = {
        ("s2", "s1"): "1",
        ("s2", "s2"): "2",
    }
    jvec = [1, 0]

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

    wanted_gamma = np.diag([1.47134, 1.079278])

    wanted_a_matrix = np.asarray(
        [
            [1, -1.079278],
            [0, 1.079278],
        ]
    )


class SequentialModelWithBacktransfer:
    params = [0.55, 0.0404, 0.11]
    compartments = ["s1", "s2"]
    matrix = {
        ("s2", "s1"): "1",
        ("s1", "s2"): "3",
        ("s2", "s2"): "2",
    }
    jvec = [1, 0]

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

    wanted_gamma = np.diag([-1.19068, -0.8872538])

    wanted_a_matrix = np.asarray(
        [
            [0.8152501, -0.8678057],
            [0.1847499, 0.8678057],
        ]
    )


class ParallelModel:
    params = [0.55, 0.0404]
    compartments = ["s1", "s2"]
    matrix = {
        ("s1", "s1"): "1",
        ("s2", "s2"): "2",
    }
    jvec = [1, 1]

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

    wanted_gamma = np.diag([1, 1])

    wanted_a_matrix = np.asarray(
        [
            [1, 0],
            [0, 1],
        ]
    )


class ParallelModelWithEquilibria:
    params = [0.55, 0.0404, 0.11, 0.02]
    compartments = ["s1", "s2"]
    matrix = {
        ("s1", "s1"): "4",
        ("s2", "s1"): "1",
        ("s2", "s2"): "2",
        ("s1", "s2"): "3",
    }
    jvec = [1, 1]

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

    wanted_gamma = np.diag([-0.940184, -1.710398])

    wanted_a_matrix = np.asarray(
        [
            [0.6543509, -0.6751081],
            [0.3456491, 1.6751081],
        ]
    )


@pytest.mark.parametrize(
    "matrix",
    [SequentialModel, SequentialModelWithBacktransfer, ParallelModel, ParallelModelWithEquilibria],
)
def test_a_matrix_general(matrix):
    params = Parameters.from_list(matrix.params)

    mat = KMatrix(label="", matrix=matrix.matrix)
    mat = fill_item(mat, None, params)

    initial_concentration = matrix.jvec

    for comp in matrix.compartments:
        assert comp in mat.involved_compartments()

    print(mat.reduced(matrix.compartments))
    assert np.array_equal(mat.reduced(matrix.compartments), matrix.wanted_array)

    print(mat.full(matrix.compartments).T)
    assert np.allclose(mat.full(matrix.compartments), matrix.wanted_full)

    print(mat.eigen(matrix.compartments)[0])
    print(mat.eigen(matrix.compartments)[1])
    vals, vec = mat.eigen(matrix.compartments)
    assert np.allclose(vals, matrix.wanted_eigen_vals)
    assert np.allclose(vec, matrix.wanted_eigen_vec)

    print(calculate_gamma(vec, initial_concentration))
    assert np.allclose(calculate_gamma(vec, initial_concentration), matrix.wanted_gamma)

    print(mat.a_matrix_general(matrix.compartments, initial_concentration))
    assert np.allclose(
        mat.a_matrix_general(matrix.compartments, initial_concentration),
        matrix.wanted_a_matrix,
    )


def test_a_matrix_sequential():
    compartments = ["s1", "s2", "s3"]
    matrix = {
        ("s2", "s1"): "1",
        ("s3", "s2"): "2",
        ("s2", "s2"): "2",
        ("s3", "s3"): "3",
    }

    params = Parameters.from_list([3, 4, 5])
    mat = KMatrix(label="", matrix=matrix)
    mat = fill_item(mat, None, params)

    initial_concentration = [1, 0, 0]

    assert not mat.is_sequential(compartments, initial_concentration)

    matrix = {
        ("s2", "s1"): "1",
        ("s2", "s2"): "2",
    }

    compartments = ["s1", "s2"]
    params = Parameters.from_list([0.55, 0.0404])
    mat = KMatrix(label="", matrix=matrix)
    mat = fill_item(mat, None, params)

    initial_concentration = [1, 0]

    print(mat.reduced(compartments))
    assert mat.is_sequential(compartments, initial_concentration)

    wanted_a_matrix = np.asarray(
        [
            [1, -1.079278],
            [0, 1.079278],
        ]
    )

    print(mat.a_matrix_sequential(compartments))
    assert np.allclose(mat.a_matrix_sequential(compartments), wanted_a_matrix)


def test_combine_matrices():
    matrix1 = {
        ("s1", "s1"): "1",
        ("s2", "s2"): "2",
    }
    mat1 = KMatrix(label="A", matrix=matrix1)
    matrix2 = {
        ("s2", "s2"): "3",
        ("s3", "s3"): "4",
    }
    mat2 = KMatrix(label="B", matrix=matrix2)

    combined = mat1.combine(mat2)

    assert combined.label == "A+B"
    assert combined.matrix[("s1", "s1")] == "1"
    assert combined.matrix[("s2", "s2")] == "3"
    assert combined.matrix[("s3", "s3")] == "4"


def test_kmatrix_ipython_rendering():
    """Autorendering in ipython"""

    matrix = {
        ("s1", "s1"): "1",
        ("s2", "s2"): "2",
    }
    kmatrix = KMatrix(label="A", matrix=matrix)

    rendered_obj = format_display_data(kmatrix)[0]

    test_markdown_str = "text/markdown"
    assert test_markdown_str in rendered_obj
    assert rendered_obj[test_markdown_str].startswith("| compartment")

    rendered_markdown_return = format_display_data(kmatrix.matrix_as_markdown())[0]

    assert test_markdown_str in rendered_markdown_return
    assert rendered_markdown_return[test_markdown_str].startswith("| compartment")
