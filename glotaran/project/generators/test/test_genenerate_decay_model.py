from yaml import dump

from glotaran.io import load_model
from glotaran.project.generators.generator import generate_parallel_model
from glotaran.project.generators.generator import generate_sequential_model


def test_generate_parallel_model(tmpdir_factory):
    nr_species = 5
    model_yaml = dump(generate_parallel_model(nr_species))
    print(model_yaml)  # noqa T001
    model_file = tmpdir_factory.mktemp("gen_par_model") / "model.yml"
    model_file.write_text(model_yaml, encoding="utf-8")

    model = load_model(model_file)

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


def test_generate_decay_model(tmpdir_factory):
    nr_species = 5
    model_yaml = dump(generate_sequential_model(nr_species))
    print(model_yaml)  # noqa T001
    model_file = tmpdir_factory.mktemp("gen_seq_model") / "model.yml"
    model_file.write_text(model_yaml, encoding="utf-8")

    model = load_model(model_file)

    print(model.validate())  # noqa T001
    assert model.valid()

    assert "initial_concentration_dataset_1" in model.initial_concentration
    initial_concentration = model.initial_concentration["initial_concentration_dataset_1"]
    assert initial_concentration.compartments == [f"species_{i}" for i in range(nr_species)]
    for i in range(nr_species):
        assert (
            initial_concentration.parameters[i].full_label == f"intitial_concentration.species_{i}"
        )

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
