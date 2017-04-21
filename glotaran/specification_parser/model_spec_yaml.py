import os
import csv
from ast import literal_eval as make_tuple

from .utils import get_keys_from_object, is_compact

from glotaran.model import (BoundConstraint,
                            EqualAreaConstraint,
                            EqualConstraint,
                            FixedConstraint,
                            InitialConcentration,
                            Parameter,
                            ParameterLeaf,
                            ZeroConstraint,
                            )


class Keys:
    BOUND = "bound"
    COMPARTMENT = "compartment"
    COMPARTMENTS = "compartments"
    COMPARTMENT_CONSTRAINTS = "compartment_constraints"
    COMPARTMENT_SCALING = "compartment_scaling"
    DATASET = "dataset"
    DATASETS = "datasets"
    EQUAL = "equal"
    EQUAL_AREA = "equal_area"
    EQUAL_AREA = "equal_area"
    EXPR = "expr"
    FIT = "fit"
    FIX = "fix"
    INITIAL_CONCENTRATION = "initial_concentration"
    INTERVALS = "intervals"
    LABEL = "label"
    MAX = "max"
    MEGACOMPLEX = "megacomplex"
    MEGACOMPLEXES = "megacomplexes"
    MEGACOMPLEX_SCALING = "megacomplex_scaling"
    MIN = "min"
    PARAMETER = "parameter"
    PARAMETERS = "parameters"
    PARAMETER_CONSTRAINTS = "parameter_constraints"
    PATH = "path"
    RANGE = "range"
    SCALING = "scaling"
    SUBBLOCKS = "sub_blocks"
    TARGET = "target"
    TO = "to"
    TYPE = 'type'
    WEIGHT = "weight"
    ZERO = "zero"


ModelParser = {}


def get_model_parser(spec):
    if spec[Keys.TYPE] in ModelParser:
        return ModelParser[spec[Keys.TYPE]](spec)
    else:
        raise Exception("Unsupported model type {}."
                        .format(spec[Keys.TYPE]))


def register_model_parser(type_name, parser):
    if not issubclass(parser, ModelSpecParser):
        raise TypeError
    ModelParser[type_name] = parser


class ModelSpecParser(object):

    def __init__(self, spec):
        self.spec = spec
        self.model = self.get_model()()

    def get_model(self):
        raise NotImplementedError

    def get_megacomplexes(self):
        raise NotImplementedError

    def get_additionals(self):
        raise NotImplementedError

    def get_dataset_descriptor(self, label, initial_concentration,
                               megacomplexes, megacomplex_scalings,
                               dataset_scaling, compartment_scalings):
        raise NotImplementedError

    def get_dataset(self, dataset_spec):
        label = dataset_spec[Keys.LABEL]
        # path = dataset_spec[DatasetKeys.PATH]
        # type = dataset_spec[Keys.TYPE]
        initial_concentration = \
            dataset_spec[Keys.INITIAL_CONCENTRATION] if \
            Keys.INITIAL_CONCENTRATION in dataset_spec else \
            None
        megacomplexes = dataset_spec[Keys.MEGACOMPLEXES]

        dataset_scaling = \
            dataset_spec[Keys.SCALING] if \
            Keys.SCALING in dataset_spec else None

        cmplx_scalings = {}
        if Keys.MEGACOMPLEX_SCALING in dataset_spec:
            for scaling in dataset_spec[Keys.MEGACOMPLEX_SCALING]:
                (cmplx, params) = \
                    get_keys_from_object(scaling, [Keys.MEGACOMPLEX,
                                                   Keys.PARAMETER])
                cmplx_scalings[cmplx] = params

        compartment_scalings = {}
        if Keys.COMPARTMENT_SCALING in dataset_spec:
            for scaling in dataset_spec[Keys.COMPARTMENT_SCALING]:
                (c, param) = get_keys_from_object(scaling, [Keys.COMPARTMENT,
                                                            Keys.PARAMETER])
                compartment_scalings[c] = param

        self.model.add_dataset(
            self.get_dataset_descriptor(label,
                                        initial_concentration,
                                        megacomplexes,
                                        cmplx_scalings,
                                        dataset_scaling,
                                        compartment_scalings,
                                        dataset_spec))

    def get_dataset_additionals(self, dataset, dataset_spec):
        raise NotImplementedError

    def get_datasets(self):
        for dataset_spec in self.spec[Keys.DATASETS]:
            self.get_dataset(dataset_spec)

    def get_parameters(self):
        params = self.spec[Keys.PARAMETERS]
        if isinstance(params, str):
            if os.path.isfile(params):
                f = open(params)
                params = f.read()
                f.close
            dialect = csv.Sniffer().sniff(params.splitlines()[0])

            reader = csv.reader(params.splitlines(), dialect)

            params = []
            for row in reader:
                params += [float(e) for e in row]
        self.model.parameter = self.get_leaf("p", params)

    def get_leaf(self, label, items):
        leaf = ParameterLeaf(label)
        for item in items:
            if isinstance(item, dict):
                label, items = list(item.items())[0]
                leaf[label] = self.get_leaf(label, items)
            elif isinstance(item, bool):
                leaf.fit = item
            else:
                leaf.add_parameter(self.get_parameter(item))
        return leaf

    def get_parameter(self, p):
        param = Parameter()

        if not isinstance(p, list):
            param.value = p
            return param

        def retrieve(filt, default):
            tmp = list(filter(filt, p))
            return tmp[0] if tmp else default

        param.value = retrieve(lambda x: isinstance(x, (int, float)) and not
                               isinstance(x, bool), 'nan')
        param.label = retrieve(lambda x: isinstance(x, str) and not
                               x == 'nan', None)
        param.vary = retrieve(lambda x: isinstance(x, bool), True)
        options = retrieve(lambda x: isinstance(x, dict), None)

        if options is not None:
            if Keys.MAX in options:
                param.max = options[Keys.MAX]
            if Keys.MIN in options:
                param.min = options[Keys.MIN]
            if Keys.EXPR in options:
                param.expr = options[Keys.EXPR]
        return param

    def get_parameter_constraints(self):
        if Keys.PARAMETER_CONSTRAINTS not in self.spec:
            return
        for constraint in self.spec[Keys.PARAMETER_CONSTRAINTS]:

            (type,) = get_keys_from_object(constraint, [Keys.TYPE])

            params = []
            if Keys.RANGE in constraint:
                params = make_tuple(constraint[Keys.RANGE])
            elif Keys.PARAMETER in constraint:
                params = constraint[Keys.PARAMETER]
            elif is_compact(constraint):
                params = constraint[1]
                if isinstance(params, str):
                    params = make_tuple(params)

            if type == Keys.FIX:
                self.model.add_parameter_constraint(
                    FixedConstraint(params))
            elif type == Keys.BOUND:
                lower = 'NaN'
                upper = 'NaN'
                if is_compact(constraint):
                    lower = constraint[2]
                    upper = constraint[3]
                else:
                    if Keys.LOWER in constraint:
                        lower = float(constraint[Keys.LOWER])
                    if Keys.UPPER in constraint:
                        upper = float(constraint[Keys.UPPER])
                self.model.add_parameter_constraint(BoundConstraint(
                    params,
                    lower=lower,
                    upper=upper))

    def get_compartment_constraints(self):
        if Keys.COMPARTMENT_CONSTRAINTS not in self.spec:
            return
        for constraint in self.spec[Keys.COMPARTMENT_CONSTRAINTS]:
            (tpe, c, intv) = get_keys_from_object(constraint,
                                                  [Keys.TYPE,
                                                   Keys.COMPARTMENT,
                                                   Keys.INTERVALS])
            intervals = []
            for interval in intv:
                intervals.append(make_tuple(interval))
            if tpe == Keys.ZERO:
                self.model.add_compartment_constraint(
                    ZeroConstraint(c, intervals))

            else:
                (target, param) = get_keys_from_object(constraint,
                                                       [Keys.TARGET,
                                                        Keys.PARAMETER],
                                                       start=3)

                if tpe == Keys.EQUAL:
                    self.model.add_compartment_constraint(
                        EqualConstraint(c, intervals, target, param))
                elif tpe == Keys.EQUAL_AREA:
                    (weight,) = get_keys_from_object(constraint,
                                                     [Keys.WEIGHT],
                                                     start=5)
                    self.model.add_compartment_constraint(
                        EqualAreaConstraint(c, intervals, target, param,
                                            weight))

    def get_initial_concentrations(self):
        if Keys.INITIAL_CONCENTRATION not in self.spec:
            return
        for concentration in self.spec[Keys.INITIAL_CONCENTRATION]:
            (label, parameter) = get_keys_from_object(concentration,
                                                      [Keys.LABEL,
                                                       Keys.PARAMETER])
            self.model.add_initial_concentration(
                InitialConcentration(label, parameter))

    def get_compartments(self):
        self.model.compartments = self.spec[Keys.COMPARTMENTS]

    def parse(self):
        self.get_compartments()
        self.get_parameters()
        self.get_parameter_constraints()
        self.get_compartment_constraints()
        self.get_initial_concentrations()
        self.get_megacomplexes()
        self.get_datasets()
        self.get_additionals()
