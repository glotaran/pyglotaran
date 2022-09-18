from glotaran.model_new.model import DEFAULT_DATASET_GROUP
from glotaran.model_new.model import Model


def test_model_create():
    m = Model.create([])()
    print(m)
    assert DEFAULT_DATASET_GROUP in m.dataset_groups

    m = Model.create([])(
        **{
            "dataset_groups": {
                "test": {"residual_function": "non_negative_least_squares", "link_clp": False}
            }
        }
    )
    print(m)
    assert DEFAULT_DATASET_GROUP in m.dataset_groups
    assert "test" in m.dataset_groups
    assert m.dataset_groups["test"].residual_function == "non_negative_least_squares"
    assert not m.dataset_groups["test"].link_clp


def test_global_item():

    m = Model.create([])(
        **{
            "weights": [
                {"datasets": ["d1", "d2"], "value": 1},
                {"datasets": ["d3"], "value": 2, "global_interval": (5, 6)},
            ]
        }
    )
    print(m)
    assert len(m.weights) == 2
    w = m.weights[0]
    assert w.datasets == ["d1", "d2"]
    assert w.value == 1
    assert w.model_interval is None
    assert w.global_interval is None

    w = m.weights[1]
    assert w.datasets == ["d3"]
    assert w.value == 2
    assert w.model_interval is None
    assert w.global_interval == (5, 6)
