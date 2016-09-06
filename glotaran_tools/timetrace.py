from glotaran_core import Dataset


class Timetrace(Dataset):
    """
    Represents a Dataset where observations are timepoints.
    """
    def __init__(self, label, channels, channel_labels, timepoints,
                 timeunit="s"):
        super(Dataset, self, label, channels, channel_labels, timepoints)
        self.timeunit = timeunit
