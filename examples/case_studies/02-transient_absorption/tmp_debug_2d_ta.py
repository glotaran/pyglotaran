# %%  # noqa: INP001
from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from pyglotaran_extras import plot_overview
from pyglotaran_extras.compat import convert

from glotaran.io import load_dataset
from glotaran.io import load_parameters
from glotaran.io import load_scheme
from glotaran.utils.json_schema import create_model_scheme_json_schema


def initialize(root_dir):
    parameters = load_parameters(root_dir / "parameters_2d_co_co2.yml")
    create_model_scheme_json_schema(root_dir / "schema_2d.json", parameters)
    scheme = load_scheme(root_dir / "scheme_2d_co_co2.yml")
    scheme.load_data(
        {
            "dataset1": load_dataset(root_dir / "data/2016co_tol.ascii"),
            "dataset2": load_dataset(root_dir / "data/2016c2o_tol.ascii"),
        }
    )
    return scheme, parameters


def optimize(scheme, parameters):
    return scheme.optimize(parameters=parameters, maximum_number_function_evaluations=7)


def plot(result):
    result_plot_ds1, _ = plot_overview(convert(result.data["dataset1"]), linlog=True)
    result_plot_ds2, _ = plot_overview(convert(result.data["dataset2"]), linlog=True)
    result_plot_ds1.show()
    result_plot_ds2.show()
    plt.show()


def save_result(result):
    folder_name = Path().resolve().name
    temp_folder = Path(tempfile.gettempdir())
    result_path = temp_folder / folder_name
    result.save(result_path, allow_overwrite=True)
    print(f"Result saved to {result_path}. \n get it while it's hot!")  # noqa: T201


if __name__ == "__main__":
    root_dir = Path(__file__).parent
    scheme, parameters = initialize(root_dir)
    result = optimize(scheme, parameters)
    save_result(result)
    plot(result)
