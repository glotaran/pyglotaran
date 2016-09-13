class Dataset(object):
    """
    Class representing a dataset for fitting.
    """
    def __init__(self, label, channels, channel_labels, observations):
        if not isinstance(channels, list):
            raise TypeError

        if not isinstance(channel_labels, list):
            raise TypeError

        if any(not isinstance(channel, list) for channel in channels):
            raise TypeError

        if any(any(not isinstance(channel_value, int) and not
                   isinstance(channel_value, float) for channel_value in
                   channel) for channel in channels):
            raise TypeError

        if not len(channels) == len(channel_labels):
            raise Exception("To few or too much labels for channels")

        if any(not len(channel) == len(observations) for channel in channels):
            raise Exception("Channels must be of same length as observations.")

        # TODO: Remove NaN, inf, etc.

        self._label = label
        self._channels = channels
        self._channel_labels = channel_labels
        self._observations = observations
        self._conncentration_scaling = None
        self._megacomplexes = []
        self._scalings = []

    def label(self):
        return self._label

    def channel_labels(self):
        return self._channel_labels

    def get_channels(self):
        return self._channels

    def get_channel(self, label):
        if label not in self._channel_labels:
            raise Exception("Non-existing channel: {}".format(label))
        return self._channels[self._channel_labels.index(label)]

    def number_of_channels(self):
        return len(self._channel_labels)

    def observations(self):
        return self._observations

    def set_concentration_scaling(self, parameter):
        if not isinstance(parameter, int):
            raise TypeError
        self._concentration_scaling = parameter

    def concentration_scaling(self):
        return self._conncentration_scaling

    def add_scaling(self, scaling):
        if not isinstance(scaling, list):
            scaling = [scaling]
        if any(not isinstance(s, Scaling) for s in scaling):
            raise TypeError

        for s in scaling:
            self._scalings.append(s)

    def scalings(self):
        return self._scalings

    def add_megacomplex(self, megacomplex):
        if not isinstance(megacomplex, list):
            megacomplex = [megacomplex]
        if any(not isinstance(m, str) for m in megacomplex):
            raise TypeError
        for m in megacomplex:
            self._megacomplexes.append(m)

    def megacomplexes(self):
        return self._megacomplexes

    def __str__(self):
        s = "Dataset '{}'\n\n".format(self.label())

        s += "\tConcentration Scaling Parameter: {}\n"\
            .format(self.concentration_scaling())

        s += "\tMegacomplexes: {}\n".format(self.megacomplexes())

        if len(self.scalings()) is not 0:
            s += "\tScalings:\n"
            for sc in self.scalings():
                s += "\t\t- {}\n".format(sc)

        return s


class Scaling(object):
    def __init__(self, megacomplexes, compartments, parameter):
        if not isinstance(megacomplexes, list):
            megacomplexes = [megacomplexes]
        if any(not isinstance(m, str) for m in megacomplexes):
            raise TypeError
        if not isinstance(compartments, list):
            compartments = [compartments]
        if any(not isinstance(c, int) for c in compartments):
            raise TypeError
        if not isinstance(parameter, int):
            raise TypeError

        self.megacomplexes = megacomplexes
        self.compartments = compartments
        self.parameter = parameter

    def __str__(self):
        return "Megacomplexes: {} Compartments: {} Parameter:{}"\
                .format(self.megacomplexes, self.compartments,
                        self.parameter)
