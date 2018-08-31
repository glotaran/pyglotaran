import numpy as np
from glotaran.model import InitialConcentration, ParameterGroup
from glotaran.models.spectral_temporal import KMatrix


def test_matrix():
    params = ParameterGroup.from_list([1, 2, 3, 0])
    mat = KMatrix("k1", {
        ('s1', 's1'): '1',
        ('s2', 's1'): '2',
        ('s2', 's2'): '1'})

    mat = mat.fill(None, params)

    con = InitialConcentration('j1', ["1", "1"]).fill(None, params)

    assert 's1' in mat.involved_compartments()
    assert 's2' in mat.involved_compartments()
    assert np.array_equal(mat.asarray(['s1', 's2']),
                          np.asarray([
                              [1, 0],
                              [2, 1]
                          ]))

    print(mat.asarray(['s1', 's2']))
    print(mat.full(['s1', 's2']))
    assert np.array_equal(mat.full(['s1', 's2']),
                          np.asarray([
                              [-3, 0],
                              [2, -1]
                          ]))

    print(mat.eigen(['s1', 's2']))
    print(mat.a_matrix(['s1', 's2'], con))
    assert False
