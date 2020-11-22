import click
from click import echo
from click import pause
from click import prompt

from . import util


@click.option(
    "--name",
    "-n",
    default=None,
    type=str,
    show_default=True,
    help="Name of the datagroup to export.",
)
@click.option(
    "--select",
    "-s",
    default=None,
    type=(str, util.VALORRANGEORLIST),
    show_default=True,
    multiple=True,
    help="Selection of data. Example --select DIMENSION_NAME VALUE "
    "where VALUE can be a scalar, a list like '[V1,V2,V3]' or a tuple like '(MIN,MAX)'",
)
@click.option(
    "--out",
    "-o",
    default=None,
    type=click.Path(dir_okay=False),
    show_default=True,
    help="Path to the output file.",
)
@click.argument("filename", type=click.Path(exists=True))
def export(filename: str, select, out: str, name: str):
    """Exports data from netCDF4 to ascii."""

    echo(f"Opening dataset at {filename}")

    dataset = util.load_dataset(filename, dtype="nc")

    if name is None:
        name = util.select_name(filename, dataset)

    stop = False
    data = dataset[name]

    for sel in select:
        data = util.select_data(data, sel[0], sel[1])

    if out is not None:
        if len(data.shape) > 2:
            pause("Cannot export data with more than 2 dimensions to ASCII")
        else:
            util.write_data(data, out)
            stop = True

    while not stop:

        echo(f"Selected dataset '{name}'.")
        echo(f"\nDataset Content\n\n{data}\n")

        choice = prompt(
            "How to proceed",
            default="exit",
            type=click.Choice(["exit", "save", "select", "change", "help"]),
        )
        if choice == "exit":
            stop = True
        elif choice == "change":
            name = util.select_name(filename, dataset)
            data = dataset[name]
        elif choice == "select":
            cont = True
            while cont:
                dims = [d for d in data.dims]
                choice = prompt(
                    "Please select a dimension or action",
                    default="back",
                    type=click.Choice(dims + ["reset", "back"]),
                )
                if choice == "back":
                    cont = False
                elif choice == "reset":
                    data = dataset[name]
                else:
                    dim = choice
                    choice = prompt(
                        "Please select a value. Type 2 values sperated by ',' to select a range.",
                        default="back",
                        type=util.VALORRANGEORLIST,
                    )
                    if choice != "back":
                        try:
                            data = util.select_data(data, dim, choice)
                        except ValueError as e:
                            echo(e, err=True)
                            continue

        elif choice == "save":
            if len(data.shape) > 2:
                pause("Cannot export data with more than 2 dimensions to ASCII")
            else:
                cont = True
                change = False
                while cont:
                    if out is None or change:
                        out = prompt(
                            "Please type in a name for the output file.",
                            default="out.csv",
                            type=click.Path(dir_okay=False),
                        )
                        change = False
                    choice = prompt(
                        f"The filepath of the output file is '{out}'. Save the data?",
                        default="yes",
                        type=click.Choice(["yes", "no", "change"]),
                    )
                    if choice == "change":
                        change = True
                    else:
                        cont = False
                        if choice == "yes":
                            util.write_data(data, out)
    echo("Good-bye, have a nice day!")
