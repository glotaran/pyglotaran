class Result(object):
    """
    Represents the results of a fit.
    """
    def best_fit_parameters(self):
        raise NotImplementedError

    def fit_error(self):
        raise NotImplementedError

    def report(self):
        """
        Returns a printable string.
        """
        raise NotImplementedError
