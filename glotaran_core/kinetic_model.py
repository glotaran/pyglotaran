from .model import Model


class KineticParameter(object):
    """
    Represents a kenetic parameter.

    Parameter
    ---------

    label = String label for the parameter
    rate = Start parameter for the rate. NaN if free parameter.
    source = Label of the source state
    target = Label of the target state, None for loss.
    positive = Indicates if parameter mus be positive.
    constrain = Tupel of lower and upper constrain.
    """
    def __init(self, label, rate, source, target=None, positive=False,
               constrain=None):
        self.label = label
        self.rate = rate
        self.source = source
        self.target = target
        self.positive = positive
        self.constrain = constrain


class Irf(object):
    """
    Represents an abstract IRF.

    Parameter
    ---------

    center: center of the irf as channel label
    center_dispersion: polynomial coefficients for the dispersion of the
        center. None for no dispersion.
    """
    def __init__(self, center, center_dispersion=None):
        self.center = center
        self.center_dispersion = center_dispersion

    def get_irf_function(self):
        raise NotImplementedError


class GaussianIrf(Irf):
    """
    Represents a simple gaussian IRF.

    Parameter
    ---------

    center: center of the irf as channel label
    width: width of the gaussian
    center_dispersion: polynomial coefficients for the dispersion of the
        center. None for no dispersion.
    width_dispersion: polynomial coefficients for the dispersion of the
        width. None for no dispersion.

    """
    def __init__(self, center, width, center_dispersion=None,
                 width_dispersion=None):
        super(Irf, self, center, center_dispersion=center_dispersion)


class KineticModel(Model):
    """
    Represents a kinetic model.

    Parameter
    ---------

    labels = One or a list labels indicating dataset(s) to be fittet with the
        model.
    kinetic_parameters = list of kinetic parameters
    irf = the irf of the model.
    parent = The model from which the model has been derived
    """
    def __init__(self, labels, kinetic_parameters, irf,
                 parent=None):
        self.kinetic_parameters = {}
        self.overwritten_parameters = []
        for k in kinetic_parameters:
            if parent is not None:
                if k.label in parent.get_kinetic_parameter_labels():
                    self.overwritten_parameters.append(k.label)
            self.kinetic_parameters[k.label] = k

        self.irf = irf

        self.parent = parent

        super(Model, self, labels)

    def derive(self, labels, kinetic_parameters, irf_parameter=None):
        """
        Derives a new model. Kinetic parameters with labels existing in the
        parent model will be freed.
        """
        return KineticModel(labels, kinetic_parameters, irf_parameter,
                            parent=self)

    def get_kinetic_parameter_labels(self):
        labels = self.kinetic_parameters.keys()
        if self.parent is not None:
            labels.append(self.parent.get_kinetic_parameter_labels())
        return set(labels)

    def get_kinetic_parameter(self, label):
        if label in self.kinetic_parameters:
            return self.kinetic_parameters[label]
        return self.parent.get_kinetic_parameter

    def get_irf(self):
        if self.irf is not None:
            return self.irf
        return self.parent.get_irf()
