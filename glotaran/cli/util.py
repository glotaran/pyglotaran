import sys
from pathlib import Path

from click import echo

import glotaran as gta


def load_model_file(filename, verbose=False):
    try:
        model = gta.read_model_from_yml_file(filename)
        if verbose:
            echo("Model parsing successfull.")
        return model
    except Exception as e:
        echo(message=f"Error parsing model file: \n\n{e}", err=True)
        sys.exit(1)


def load_parameter_file(filename, verbose=False):
    try:
        parameter = gta.read_parameter_from_yml_file(filename)
        if verbose:
            echo("Parameter parsing successfull.")
        return parameter
    except Exception as e:
        echo(message=f"Error parsing parameter file: \n\n{e}", err=True)
        sys.exit(1)


file_readers = {
    'ascii': gta.io.read_ascii_time_trace,
    'sdt': gta.io.read_sdt_data,
}


def load_dataset(path):
    path = Path(path)

    if path.suffix[1:] not in file_readers:
        echo(f"Unknown file type '{path.suffix[1:]}'."
             f"Supported file types are {list(file_readers.keys())}.", err=True)
        sys.exit(1)

    try:
        dataset = file_readers[path.suffix[1:]](path)
        echo("Dataset loading successfull.")
        return dataset
    except Exception as e:
        echo(message=f"Error loading dataset file: \n\n{e}", err=True)
        sys.exit(1)
