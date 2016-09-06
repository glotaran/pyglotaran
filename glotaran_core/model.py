class Model(object):
    """
    Model represents a global analysis model.

    It contains parameters and labels.

    The labels are used to link a model to dataset in the fitting.
    """
    def __init__(self, labels, rate_constants, contrains):
        if not isinstance(labels, list):
            labels = [labels]
        # TODO: maybe allow non numeric labels 
        if any(not isinstance(label, basestring) for label in labels):
            raise TypeError

    def 
