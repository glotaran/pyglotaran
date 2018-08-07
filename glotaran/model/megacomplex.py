class Megacomplex(object):
    """
    A megacomplex has a label.
    """
    def __init__(self, label):
        if not isinstance(label, str):
            raise TypeError
        self.label = label

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    def __str__(self):
        return f"### _{self.label}_\n"
