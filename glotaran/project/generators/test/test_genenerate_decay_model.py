import pytest

from glotaran.project.generators.generator import generate_model


@pytest.mark.parametrize("megacomplex_type", ["parallel", "sequential"])
@pytest.mark.parametrize("irf", [True, False])
@pytest.mark.parametrize("spectral", [True, False])
def test_generate_parallel_model(megacomplex_type: str, irf: bool, spectral: bool):
    nr_compartments = 5
    expected_compartments = [f"species_{i+1}" for i in range(nr_compartments)]
    model_type = f"spectral_decay_{megacomplex_type}" if spectral else f"decay_{megacomplex_type}"
    model = generate_model(
        generator_name=model_type,
        generator_arguments={
            "nr_compartments": nr_compartments,
            "irf": irf,
        },
    )
    print(model)

    assert (
        f"megacomplex_{megacomplex_type}_decay" in model.megacomplex  # type:ignore[attr-defined]
    )
    megacomplex = model.megacomplex[  # type:ignore[attr-defined]
        f"megacomplex_{megacomplex_type}_decay"
    ]
    assert megacomplex.type == f"decay-{megacomplex_type}"
    assert megacomplex.compartments == expected_compartments
    assert [r.full_label for r in megacomplex.rates] == [
        f"rates.species_{i+1}" for i in range(nr_compartments)
    ]

    assert "dataset_1" in model.dataset  # type:ignore[attr-defined]
    dataset = model.dataset["dataset_1"]  # type:ignore[attr-defined]
    assert dataset.megacomplex == [f"megacomplex_{megacomplex_type}_decay"]

    if spectral:
        assert "megacomplex_spectral" in model.megacomplex  # type:ignore[attr-defined]
        megacomplex = model.megacomplex["megacomplex_spectral"]  # type:ignore[attr-defined]
        assert expected_compartments == list(megacomplex.shape.keys())
        expected_shapes = [f"shape_species_{i+1}" for i in range(nr_compartments)]
        assert expected_shapes == list(megacomplex.shape.values())

        for i, shape in enumerate(expected_shapes):
            assert shape in model.shape  # type:ignore[attr-defined]
            assert model.shape[shape].type == "gaussian"  # type:ignore[attr-defined]
            assert (
                model.shape[shape].amplitude.full_label  # type:ignore[attr-defined]
                == f"shapes.species_{i+1}.amplitude"
            )
            assert (
                model.shape[shape].location.full_label  # type:ignore[attr-defined]
                == f"shapes.species_{i+1}.location"
            )
            assert (
                model.shape[shape].width.full_label  # type:ignore[attr-defined]
                == f"shapes.species_{i+1}.width"
            )
            assert dataset.global_megacomplex == ["megacomplex_spectral"]

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
