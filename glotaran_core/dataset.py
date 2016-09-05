class Dataset(object):
    """
    Abstract class representing a dataset for fitting.
    """
    def __init__(self, name):
        self.name = name

    def wavenumbers(self):
        try:
            wn = []
            for wl in self.wavelengths():
                wn.append(10000000/wl)
            return wn
        except:
            raise NotImplementedError

    def wavelengths(self):
        try:
            wl = []
            for wn in self.wavenumbers():
                wl.append(10000000/wn)
            return wl
        except:
            raise NotImplementedError

    def timepoints(self):
        raise NotImplementedError

    def data(self):
        raise NotImplementedError
