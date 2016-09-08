

class Model(object):
    """
    Model represents a global analysis model.

    The labels are used to link a model to dataset in the fitting.
    """
    def __init__(self, labels):
        if not isinstance(labels, list):
            labels = [labels]

    def labels(self):
        return self._labels
