from glotaran.model import ParameterGroup


def test_param_array():
    params = """
    - 5
    - 4
    - 3
    - 2
    - 1
    """

    params = ParameterGroup.from_yaml(params)

    assert len(list(params.all())) is 5

    assert [p.label for p in params.all()] == [f"{i}" for i in range(1, 6)]
    assert [p.value for p in params.all()] == list(range(1, 6))[::-1]


def test_param_label():
    params = """
    - ["5", 1]
    - ["4", 2]
    - ["3", 3]
    """

    params = ParameterGroup.from_yaml(params)

    assert len(list(params.all())) is 3
    assert [p.label for p in params.all()] == [f"{i}" for i in range(5, 2, -1)]
    assert [p.value for p in params.all()] == list(range(1, 4))


def test_param_options():
    params = """
    - ["5", 1, {fit: false, min: -1, max: 1, vary: false}]
    - ["6", 4e2, {fit: true, min: -7e2, max: 8e2, vary: true}]
    - ["7", 2e4]
    """

    params = ParameterGroup.from_yaml(params)

    assert params.get("5").value == 1.0
    assert not params.get("5").fit
    assert params.get("5").min == -1
    assert params.get("5").max == 1
    assert not params.get("5").vary

    assert params.get("6").value == 4e2
    assert params.get("6").fit
    assert params.get("6").min == -7e2
    assert params.get("6").max == 8e2
    assert params.get("6").vary

    assert params.get("7").value == 2e4
    assert params.get("7").fit
    assert params.get("7").min == float('-inf')
    assert params.get("7").max == float('inf')
    assert params.get("7").vary


def test_nested_param_list():
    params = """
    kinetic:
        - 3
        - 4
        - 5
    j:
        - 7
        - 8
    """

    params = ParameterGroup.from_yaml(params)

    assert len(list(params.all())) is 5
    group = params['kinetic']
    assert len(list(group.all())) is 3
    assert [p.label for p in group.all()] == [f"{i}" for i in range(1, 4)]
    assert [p.value for p in group.all()] == list(range(3, 6))
    group = params['j']
    assert len(list(group.all())) is 2
    assert [p.label for p in group.all()] == [f"{i}" for i in range(1, 3)]
    assert [p.value for p in group.all()] == list(range(7, 9))


def test_nested_param_group():
    params = """
    kinetic:
        j:
            - 7
            - 8
            - 9
    """

    params = ParameterGroup.from_yaml(params)
    assert len(list(params.all())) is 3
    group = params['kinetic']
    assert len(list(group.all())) is 3
    group = group['j']
    assert len(list(group.all())) is 3
    assert [p.label for p in group.all()] == [f"{i}" for i in range(1, 4)]
    assert [p.value for p in group.all()] == list(range(7, 10))
