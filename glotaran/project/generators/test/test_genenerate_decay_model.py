import pytest

from glotaran.project.generators.generator import generate_model


@pytest.mark.parametrize("megacomplex_type", ["parallel", "sequential"])
@pytest.mark.parametrize("irf", [True, False])
def test_generate_parallel_model(megacomplex_type: str, irf: bool):
    nr_compartments = 5
    model = generate_model(
        f"decay-{megacomplex_type}",
        **{"nr_compartments": nr_compartments, "irf": irf},  # type:ignore[arg-type]
    )
    print(model)  # T001

    assert (
        f"megacomplex_{megacomplex_type}_decay" in model.megacomplex  # type:ignore[attr-defined]
    )
    megacomplex = model.megacomplex[  # type:ignore[attr-defined]
        f"megacomplex_{megacomplex_type}_decay"
    ]
    assert megacomplex.type == f"decay-{megacomplex_type}"
    assert megacomplex.compartments == [f"species_{i+1}" for i in range(nr_compartments)]
    assert [r.full_label for r in megacomplex.rates] == [
        f"decay.species_{i+1}" for i in range(nr_compartments)
    ]

    assert "dataset_1" in model.dataset  # type:ignore[attr-defined]
    dataset = model.dataset["dataset_1"]  # type:ignore[attr-defined]
    assert dataset.megacomplex == [f"megacomplex_{megacomplex_type}_decay"]
    if irf:
        assert dataset.irf == "gaussian_irf"
        assert "gaussian_irf" in model.irf  # type:ignore[attr-defined]
        assert (
            model.irf["gaussian_irf"].center.full_label  # type:ignore[attr-defined]
            == "irf.center"
        )
        assert (
            model.irf["gaussian_irf"].width.full_label == "irf.width"  # type:ignore[attr-defined]
        )
