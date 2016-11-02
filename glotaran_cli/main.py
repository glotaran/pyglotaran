import click
from click import echo
from glotaran_tools.specification_parser import parse_file


@click.group()
def glotaran():
    print("Hello glotaran")


@click.command()
@click.argument('filename')
def parse(filename):
    echo('Parsing file: {}\n\n'.format(filename))
    model = parse_file(filename)

    echo(model)

glotaran.add_command(parse)

if __name__ == '__main__':
    glotaran()
