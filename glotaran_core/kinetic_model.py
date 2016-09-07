from model import Model


class KineticParameter(object):
    def __init(self, label, start, positive=False, constrain=None):
        self.label = label
        self.start = start
        self.positive = positive
        self.constrain = constrain


class IrfParameter(object):
    def __init__(self, start, start_center):
        self.start = start
        self.start_center = start_center


class KineticModel(Model):
    def __init__(self, labels, kinetic_parameters, irf_parameter,
                 parent=None):
        self.kinetic_parameters = {}
        for k in kinetic_parameters:
            self.kinetic_parameters[k.label] = k

        self.irf_parameter = irf_parameter

        super(Model, self, labels)

    def derive(self, labels, kinetic_parameters, irf_parameter=None):
        return KineticModel(labels, kinetic_parameters, irf_parameter,
                            parent=self)

    def get_kinetic_parameter_labels(self):
        labels = self.kinetic_parameters.keys()
        if self.parent is not None:
            labels.append(self.parent.get_kinetic_parameter_labels())
        return set(labels)

    def get_irf_parameter_labels(self):
        labels = self.irf_parameters.keys()
        if self.parent is not None:
            labels.append(self.parent.get_irf_parameter_labels())
        return set(labels)

    def get_kinetic_parameter(self, label):
        if label in self.kinetic_parameters:
            return self.kinetic_parameters[label]
        return self.parent.get_kinetic_parameter

    def get_irf_parameter(self):
        if self.irf_parameter is not None:
            return self.irf_parameter
        return self.parent.get_irf_parameter()
