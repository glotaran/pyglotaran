class Megacomplex(object):
    """
    A mega complex has a label and an initial concentration vector.
    """
    def __init__(self, label, initial_concentration):
        if not isinstance(label, str) or not isinstance(initial_concentration,
                                                        list):
            raise TypeError
        if any(not isinstance(c, int) for c in initial_concentration):
            raise TypeError
        self._label = label
        self._initial_concentration = initial_concentration

    def label(self):
        return self._label

    def initial_concentration(self):
        return self._initial_concentration

    def __str__(self):
        return "Label: {}\nInitial Concentration:{}"\
          .format(self._label, self._initial_concentration)
