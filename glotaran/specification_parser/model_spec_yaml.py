import os
import csv
from ast import literal_eval as make_tuple

from .utils import get_keys_from_object, is_compact

from glotaran.model import (create_parameter_list,
                            BoundConstraint,
                            EqualAreaConstraint,
                            EqualConstraint,
                            FixedConstraint,
                            InitialConcentration,
                            ParameterBlock,
                            Relation,
                            ZeroConstraint,
                            )


class Keys:
    BOUND = "bound"
    COMPARTMENTS = "compartments"
    COMPARTMENT_SCALING = "compartment_scaling"
    COMPARTMENT_CONSTRAINTS = "compartment_constraints"
    DATASET = "dataset"
    DATASETS = "datasets"
    DATASET_SCALING = "dataset_scaling"
    EQUAL_AREA = "equal_area"
    FIX = "fix"
    INITIAL_CONCENTRATION = "initial_concentration"
    LABEL = "label"
    MEGACOMPLEXES = "megacomplexes"
    MEGACOMPLEX_SCALING = "megacomplex_scaling"
    PARAMETER = "parameter"
    PARAMETERS = "parameters"
    PARAMETER_BLOCK = "parameter_block"
    PARAMETER_CONSTRAINTS = "parameter_constraints"
    PATH = "path"
    RANGE = "range"
    RELATIONS = "relations"
    SUBBLOCKS = "sub_blocks"
    TARGET = "target"
    TO = "to"
    TYPE = 'type'
    WEIGHT = "weight"

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
            dataset_spec[Keys.DATASET_SCALING] if \
            Keys.DATASET_SCALING in dataset_spec else None

        cmplx_scalings = {}
        if Keys.MEGACOMPLEX_SCALING in dataset_spec:
            for scaling in dataset_spec[Keys.MEGACOMPLEX_SCALING]:
                (cmplx, params) = \
                    get_keys_from_object(scaling, [Keys.MEGACOMPLEX,
                                                   Keys.PARAMETER])
                cmplx_scalings[cmplx] = params

        compartment_scalings = {}
        if Keys.COMPARTMENT_SCALING in dataset_spec:
            for scaling in dataset_spec[Keys.COMPARTEMENT_SCALING]:
                (c, param) = get_keys_from_object(scaling, [Keys.COMPARTMENT,
                                                            Keys.PARAMETER])
                compartment_scalings[c] = params

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

    def get_parameter(self):
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
        plist = create_parameter_list(params)
        self.model.parameter = plist

    def get_parameter_blocks(self):
        if Keys.PARAMETER_BLOCK not in self.spec:
            return
        for block in self.spec[Keys.PARAMETER_BLOCK]:
            (label, params, sub_blocks) = get_keys_from_object(block,
                                                               [Keys.LABEL,
                                                                Keys.PARAMETER,
                                                                Keys.SUBBLOCKS,
                                                                ]
                                                               )
            self.model.parameter_blocks[label] = ParameterBlock(label, params,
                                                                sub_blocks)

    def get_parameter_constraints(self):
        if Keys.PARAMETER_CONSTRAINTS not in self.spec:
            return
        for constraint in self.spec[Keys.PARAMETER_CONSTRAINTS]:

            (type) = get_keys_from_object(constraint, [Keys.Type])

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
            (type, c, intv) = get_keys_from_object(constraint,
                                                   [Keys.Type,
                                                    Keys.COMPARTMENT,
                                                    Keys.INTERVALS])
            intervals = []
            for interval in intv:
                intervals.append(make_tuple(interval))
            if type == Keys.ZERO:
                self.model.add_compartment_constraint(
                    ZeroConstraint(c, intervals))

            else:
                (target, param) = get_keys_from_object(constraint,
                                                       [Keys.TARGET,
                                                        Keys.PARAMETER],
                                                       start=3)

                if type == Keys.EQUAL:
                    self.model.add_compartment_constraint(
                        EqualConstraint(c, intervals, target, param))
                elif type == Keys.EQUAL_AREA:
                    (weight) = get_keys_from_object(constraint,
                                                    [Keys.WEIGHT],
                                                    start=5)
                    self.model.add_compartment_constraint(
                        EqualAreaConstraint(c, intervals, target, param,
                                            weight))

    def get_relations(self):
        if Keys.RELATIONS not in self.spec:
            return

        for relation in self.spec[Keys.RELATIONS]:
            (params, to) = get_keys_from_object(relation[Keys.PARAMETER,
                                                         Keys.TO])
            self.model.add_relation(Relation(params, to))

    def get_initial_concentrations(self):
        if Keys.INITIAL_CONCENTRATION not in self.spec:
            return
        for concentration in self.spec[Keys.INITIAL_CONCENTRATIONS]:
            (label, parameter) = get_keys_from_object(concentration,
                                                      [Keys.LABEL,
                                                       Keys.PARAMETER])
            self.model.add_initial_concentration(
                InitialConcentration(label, parameter))

    def get_compartments(self):
        self.model.compartments = self.spec[Keys.COMPARTMENTS]

    def parse(self):
        self.get_compartments()
        self.get_parameter()
        self.get_parameter_constraints()
        self.get_compartment_constraints()
        self.get_relations()
        self.get_initial_concentrations()
        self.get_megacomplexes()
        self.get_datasets()
        self.get_additionals()
