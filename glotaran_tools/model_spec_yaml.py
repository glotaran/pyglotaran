import yaml
import os
import csv
from ast import literal_eval as make_tuple
from glotaran_core import (KineticModel,
                           KineticMegacomplex,
                           create_parameter_list,
                           KMatrix
                           )


models = {
    "kinetic": KineticModel
}


def get_kinetic_megacomplexes(model, spec):
    if "k_matrices" not in spec:
        raise Exception("No k-matrices defined")
    for km in spec['k_matrices']:
        print(km['matrix'])
        m = {}
        for i in km['matrix']:
            m[make_tuple(i)] = km['matrix'][i]
        model.add_k_matrix(KMatrix(km['label'], m))
    for cmplx in spec['megacomplexes']:
        l = 'label'
        km = 'k_matrices'
        if isinstance(cmplx, list):
            l = 0
            km = 1
        model.add_megacomplex(KineticMegacomplex(cmplx[l],
                                                 cmplx[km]))


megacomplex_parser = {
    "kinetic": get_kinetic_megacomplexes
}


def parse_file(fname):
    if not os.path.isfile(fname):
        raise Exception("File does not exist.")

    f = open(fname)
    spec = load(f)
    f.close
    return parse_spec(spec)


def load(s):
    return yaml.load(s)


def parse_spec(spec):

    model = get_model(spec)

    get_parameter(model, spec)

    megacomplex_parser[spec['type']](model, spec)

    return model


def get_model(spec):
    if spec['type'] in models:
        return models[spec['type']]()
    else:
        raise Exception("Unsupported modeltype {}."
                        .format(spec['type']))


def get_parameter(model, spec):
    params = spec['parameters']
    if isinstance(params, str):
        if os.path.isfile(params):
            f = open(params)
            params = f.read()
            f.close
        dialect = csv.Sniffer().sniff(params.splitlines()[0])

        reader = csv.reader(params.splitlines(), dialect)

        params = []
        for row in reader:
            print(row)
            params += [float(e) for e in row]
    plist = create_parameter_list(params)
    model.add_parameter(plist)


def parse_kinetic(self):
    params = {}
    for par in self._spec['parameter']:
        print(par)
        params[par['label']] = par
    for megacmplx in self._spec['megacomplexes']:
        print(megacmplx)
        print(megacmplx['k_matrix'])
