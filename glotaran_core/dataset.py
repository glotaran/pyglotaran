class Dataset(object):
    """
    Class representing a dataset for fitting.
    """
    def __init__(self, label, channels, channel_labels, observations):
        if isinstance(channels, list):
            raise TypeError

        if not isinstance(channel_labels, list):
            raise TypeError

        # TODO: Maybe allow non-string labels
        if any(not isinstance(label, basestring) for label in channel_labels):
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
        self._observations

    def label(self):
        return self._label

    def channel_labels(self):
        return self._channel_labels

    def get_channel(self, label):
        if label not in self._channel_labels:
            raise Exception("Non-existing channel: {}".format(label))
        return self._channels[self._channel_labels.index(label)]

    def number_of_channels(self):
        return len(self._channel_labels)

    def observations(self):
        return self._observations
