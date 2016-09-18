class Megacomplex(object):
    """
    A mega complex has a label.
    """
    def __init__(self, label):
        if not isinstance(label, str):
            raise TypeError
        self._label = label

    def label(self):
        return self._label

    def __str__(self):
        return "Label: {}"\
          .format(self._label)
