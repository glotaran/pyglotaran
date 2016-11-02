class Dataset(object):

    def channels(self):
        raise NotImplementedError

    def get_channels_for_range(self, range):
        raise NotImplementedError

    def number_of_channels(self):
        return len(self.channels)

    @property
    def independent_axies(self):
        return self._observations

    @independent_axies.setter
    def independet_axies(self, independent_axies):
        if not isinstance(independent_axies, list):
            independent_axies = [independent_axies]
        if any(not isinstance(val, (int, float)) for val in independent_axies):
            raise ValueError("Non numerical independent axies")
        self._indpendent_axies = independent_axies
