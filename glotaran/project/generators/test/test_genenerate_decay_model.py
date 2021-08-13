import pytest
from yaml import dump

from glotaran.io import load_model
from glotaran.project.generators.generator import generate_parallel_model
from glotaran.project.generators.generator import generate_sequential_model


def test_generate_parallel_model():
    nr_species = 5
    model_yaml = dump(generate_parallel_model(nr_species))
    print(model_yaml)  # noqa T001

    model = load_model(model_yaml, format_name="yml_str")

    assert model.valid()

    assert "initial_concentration_dataset_1" in model.initial_concentration
    initial_concentration = model.initial_concentration["initial_concentration_dataset_1"]
    assert initial_concentration.compartments == [f"species_{i+1}" for i in range(nr_species)]
    for i in range(nr_species):
        assert (
            initial_concentration.parameters[i].full_label
            == f"intitial_concentration.species_{i+1}"
        )

    assert "k_matrix_parallel" in model.k_matrix
    k_matrix = model.k_matrix["k_matrix_parallel"]
    for i, (k, v) in enumerate(k_matrix.matrix.items()):
        assert k == (f"species_{i+1}", f"species_{i+1}")
        assert v.full_label == f"decay.species_{i+1}"

    assert "dataset_1" in model.dataset
    dataset = model.dataset["dataset_1"]
    assert dataset.initial_concentration == "initial_concentration_dataset_1"
    assert dataset.megacomplex == ["megacomplex_parallel_decay"]


@pytest.mark.parametrize("irf", [True, False])
def test_generate_decay_model(irf):
    nr_species = 5
    model_yaml = dump(generate_sequential_model(nr_species, irf=irf))
    print(model_yaml)  # noqa T001

    model = load_model(model_yaml, format_name="yml_str")

    print(model.validate())  # noqa T001
    assert model.valid()

    assert "initial_concentration_dataset_1" in model.initial_concentration
    initial_concentration = model.initial_concentration["initial_concentration_dataset_1"]
    assert initial_concentration.compartments == [f"species_{i}" for i in range(nr_species)]
    assert initial_concentration.parameters[0].full_label == "initial_concentration.1"
    for i in range(1, nr_species):
        assert initial_concentration.parameters[i].full_label == "initial_concentration.0"

    assert "k_matrix_sequential" in model.k_matrix
    k_matrix = model.k_matrix["k_matrix_sequential"]
    for i, (k, v) in enumerate(k_matrix.matrix.items()):
        if i < len(k_matrix.matrix) - 1:
            assert k == (f"species_{i+2}", f"species_{i+1}")
        else:
            assert k == (f"species_{i+1}", f"species_{i+1}")
        assert v.full_label == f"decay.species_{i+1}"

    assert "dataset_1" in model.dataset
    dataset = model.dataset["dataset_1"]
    assert dataset.initial_concentration == "initial_concentration_dataset_1"
    assert dataset.megacomplex == ["megacomplex_parallel_decay"]

    if irf:
        assert dataset.irf == "gaussian_irf"
        assert "gaussian_irf" in model.irf
        irf = model.irf["gaussian_irf"]
        assert irf.center.full_label == "irf.center"
        assert irf.width.full_label == "irf.width"
