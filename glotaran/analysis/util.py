import numpy as np


def create_svd(name, dataset, model):
    l, v, r = np.linalg.svd(dataset[name])

    dataset[f"{name}_left_singular_vectors"] = (
        (model.model_dimension, "left_singular_value_index"),
        l,
    )

    dataset[f"{name}_right_singular_vectors"] = (
        ("right_singular_value_index", model.global_dimension),
        r,
    )

    dataset[f"{name}_singular_values"] = (("singular_value_index"), v)
