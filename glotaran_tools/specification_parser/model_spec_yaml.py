import os
import csv
from ast import literal_eval as make_tuple
from glotaran_core.model import (create_parameter_list,
                                 Dataset,
                                 DatasetScaling,
                                 FixedConstraint,
                                 BoundConstraint,
                                 ZeroConstraint,
                                 EqualConstraint,
                                 EqualAreaConstraint,
                                 Megacomplex,
                                 MegacomplexScaling,
                                 Relation,
                                 InitialConcentration)


class ModelKeys:
    DATASETS = "datasets"
    PARAMETER = "parameter"
    MEGACOMPLEXES = "megacomplexes"
    COMPARTMENT_CONSTRAINTS = "compartment_constraints"
    PARAMETER_CONSTRAINTS = "parameter_constraints"
    RELATIONS = "relations"
    INITIAL_CONCENTRATIONS = "initial_concentrations"
    DATASET = "dataset"
    TYPE = 'type'
    LABEL = "label"
    RANGE = "range"


class ParameterConstraintTypes:
    FIX = "fix"
    BOUND = "bound"


class BoundParameterConstraintKeys:
    LOWER = "lower"
    UPPER = "upper"


class CompartmentConstraintTypes:
    ZERO = "zero"
    EQUAL = "equal"
    EQUAL_AREA = "equal_area"


class CompartmentConstraintKeys:
    INTERVALS = "intervals"
    COMPARTMENT = "compartment"
    TARGET = "target"
    WEIGHT = "weight"


class RelationKeys:
    TO = "to"


class DatasetKeys:
    PATH = "path"
    DATASET_SCALING = "dataset_scaling"
    MEGACOMPLEX_SCALING = "megacomplex_scaling"
    INITIAL_CONCENTRATION = "initial_concentration"
    COMPARTEMENTS = "compartments"

ModelParser = {}


def get_model_parser(spec):
    if spec[ModelKeys.TYPE] in ModelParser:
        return ModelParser[spec[ModelKeys.TYPE]](spec)
    else:
        raise Exception("Unsupported model type {}."
                        .format(spec[ModelKeys.TYPE]))


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
                               dataset, dataset_scaling):
        raise NotImplementedError

    def get_dataset(self, dataset_spec):
        label = dataset_spec[ModelKeys.LABEL]
        path = dataset_spec[DatasetKeys.PATH]
        type = dataset_spec[ModelKeys.TYPE]
        try:
            initial_concentration = dataset_spec[DatasetKeys.INITIAL_CONCENTRATION]
        except:
            initial_concentration = []
        megacomplexes = dataset_spec[ModelKeys.MEGACOMPLEXES]

        try:
            dataset_scaling = DatasetScaling(dataset_spec[DatasetKeys.DATASET_SCALING])
        except:
            dataset_scaling = None

        mss = []
        try:
            for ms in dataset_spec[DatasetKeys.MEGACOMPLEX_SCALING]:
                compact = is_compact(ms)
                mc = ModelKeys.MEGACOMPLEXES
                cp = DatasetKeys.COMPARTEMENTS
                pm = ModelKeys.PARAMETER
                if compact:
                    mc = 0
                    cp = 1
                    pm = 2
                mss.append(MegacomplexScaling(ms[mc], ms[cp], ms[pm]))
        except:
            pass

        self.model.add_dataset(
            self.get_dataset_descriptor(label,
                                        initial_concentration,
                                        megacomplexes,
                                        mss,
                                        Dataset(),
                                        dataset_scaling,
                                        dataset_spec))

    def get_dataset_additionals(self, dataset, dataset_spec):
        raise NotImplementedError

    def get_datasets(self):
        for dataset_spec in self.spec[ModelKeys.DATASETS]:
            self.get_dataset(dataset_spec)

    def get_parameter(self):
        params = self.spec[ModelKeys.PARAMETER]
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

    def get_parameter_constraints(self):
        if ModelKeys.PARAMETER_CONSTRAINTS not in self.spec:
            return
        for constraint in self.spec[ModelKeys.PARAMETER_CONSTRAINTS]:
            compact = is_compact(constraint)

            tp = ModelKeys.TYPE
            if compact:
                tp = 0

            params = []
            if ModelKeys.RANGE in constraint:
                params = make_tuple(constraint[ModelKeys.RANGE])
            elif ModelKeys.PARAMETER in constraint:
                params = constraint[ModelKeys.PARAMETER]
            elif compact:
                params = constraint[1]
                if isinstance(params, str):
                    params = make_tuple(params)

            if constraint[tp] == ParameterConstraintTypes.FIX:
                self.model.add_parameter_constraint(
                    FixedConstraint(params))
            elif constraint[tp] == ParameterConstraintTypes.BOUND:
                lower = float('nan')
                upper = float('nan')
                if compact:
                    lower = float(constraint[2])
                    upper = float(constraint[3])
                else:
                    if BoundParameterConstraintKeys.LOWER in constraint:
                        lower = float(
                            constraint[BoundParameterConstraintKeys.LOWER])
                    if BoundParameterConstraintKeys.UPPER in constraint:
                        upper = float(
                            constraint[BoundParameterConstraintKeys.UPPER])
                self.model.add_parameter_constraint(BoundConstraint(
                    params,
                    lower=lower,
                    upper=upper))

    def get_compartment_constraints(self):
        if ModelKeys.COMPARTMENT_CONSTRAINTS not in self.spec:
            return
        for constraint in self.spec[ModelKeys.COMPARTMENT_CONSTRAINTS]:
            compact = is_compact(constraint)
            tp = ModelKeys.TYPE
            cp = CompartmentConstraintKeys.COMPARTMENT
            it = CompartmentConstraintKeys.INTERVALS
            if compact:
                tp = 0
                cp = 1
                it = 2

            intervals = []
            for interval in constraint[it]:
                intervals.append(make_tuple(interval))
            if constraint[tp] == CompartmentConstraintTypes.ZERO:
                self.model.add_compartment_constraint(
                    ZeroConstraint(constraint[cp], intervals))

            else:
                tg = CompartmentConstraintKeys.TARGET
                par = ModelKeys.PARAMETER
                wg = CompartmentConstraintKeys.WEIGHT
                if compact:
                    tg = 3
                    par = 4
                    wg = 5

                if constraint[tp] == CompartmentConstraintTypes.EQUAL:
                    self.model.add_compartment_constraint(
                        EqualConstraint(constraint[cp], intervals,
                                        constraint[tg], constraint[par]))

                elif constraint[tp] == CompartmentConstraintTypes.EQUAL_AREA:
                    self.model.add_compartment_constraint(
                        EqualAreaConstraint(constraint[cp], intervals,
                                            constraint[tg], constraint[par],
                                            constraint[wg]))

    def get_relations(self):
        if ModelKeys.RELATIONS not in self.spec:
            return

        for relation in self.spec[ModelKeys.RELATIONS]:
            compact = is_compact(relation)

            par = ModelKeys.PARAMETER
            to = RelationKeys.TO

            if compact:
                par = 0
                to = 1

            self.model.add_relation(Relation(relation[par], relation[to]))

    def get_initial_concentrations(self):
        for concentration in self.spec[ModelKeys.INITIAL_CONCENTRATIONS]:
            compact = is_compact(concentration)

            lb = ModelKeys.LABEL
            par = ModelKeys.PARAMETER
            if compact:
                lb = 0
                par = 1

            self.model.add_initial_concentration(
                InitialConcentration(concentration[lb], concentration[par]))

    def parse(self):
        self.get_parameter()
        self.get_parameter_constraints()
        self.get_compartment_constraints()
        self.get_relations()
        self.get_initial_concentrations()
        self.get_megacomplexes()
        self.get_datasets()
        self.get_additionals()


def is_compact(element):
    return isinstance(element, list)
