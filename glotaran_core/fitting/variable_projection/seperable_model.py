class SeperableModel(object):
    def parameter(self):
        raise NotImplementedError

    def c_matrix(self, parameter, **kwargs):
        raise NotImplementedError

    def e_matrix(self, parameter):
        raise NotImplementedError
