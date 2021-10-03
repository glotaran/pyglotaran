import numpy as np

from glotaran.parameter.parameter_group import ParameterGroup
from glotaran.parameter.parameter_history import ParameterHistory


def test_parameter_history():
    group0 = ParameterGroup.from_list([["1", 1], ["2", 4]])
    group1 = ParameterGroup.from_list([["1", 2], ["2", 5]])
    group2 = ParameterGroup.from_list([["1", 3], ["2", 6]])

    history = ParameterHistory()

    history.append(group0)

    assert history.parameter_labels == ["1", "2"]

    assert history.number_of_records == 1
    assert all(history.get_parameters(0) == [1, 4])

    history.append(group1)

    assert history.number_of_records == 2
    assert all(history.get_parameters(1) == [2, 5])

    history.append(group2)

    assert history.number_of_records == 3
    assert all(history.get_parameters(2) == [3, 6])

    df = history.to_dataframe()

    assert all(df.columns == history.parameter_labels)
    assert np.all(df.values == history.parameters)

    group2.set_from_history(history, 0)

    assert group2.get("1") == 1
    assert group2.get("2") == 4
