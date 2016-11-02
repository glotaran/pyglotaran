from glotaran_core import KineticDataset


class Timetrace(KineticDataset):
    """
    Represents a Dataset where observations are timepoints.
    """
    def __init__(self, label, channels, channel_labels, timepoints,
                 timeunit="s"):
        super(Timetrace, self).__init__(label, channels, channel_labels,
                                        timepoints)
        self.timeunit = timeunit
