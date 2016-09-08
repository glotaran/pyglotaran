from .spectral_timetrace import SpectralTimetrace, SpectralUnit
import os.path
import csv


class ExplicitFile(object):
    """
    Abstract representing either a time- or wavlength-explicit file.
    """
    def __init__(self, file):
        self._file = file

    def get_explicit_axies(self):
        raise NotImplementedError

    def set_explicit_axies(self, axies):
        raise NotImplementedError

    def get_implicit_axies(self):
        raise NotImplementedError

    def get_data_row(self, index):
        raise NotImplementedError

    def add_data_row(self, row):
        raise NotImplementedError

    def get_format_name(self):
        raise NotImplementedError

    def write(self, dataset, overwrite=False, comment=""):
        if not isinstance(dataset, SpectralTimetrace):
            raise TypeError

        self._dataset = dataset

        comment = comment.splitlines()
        while len(comment) < 2:
            comment.append("")

        if os.path.isfile(self._file) and not overwrite:
            raise Exception("File already exist.")

        f = open(self._file, "w")

        f.write(comment[0])
        f.write(comment[1])

        f.write(self.get_format_name())

        f.write("Intervalnr {}".format(len(self.get_explicit_axies())))

        datawriter = csv.writer(f, delimiter='\t')

        datawriter.writerow(self.get_explicit_axies())

        for i in range(len(self.get_implicit_axies())):
            datawriter.writerow(self.get_data_row(i)
                                .prepend(self.get_implicit_axies()[i]))

        f.close()

    def read(self, label, spectral_unit=SpectralUnit.nm, time_unit="s"):
        if not os.path.isfile(self._file):
            raise Exception("File does not exist.")

        f = open(self._file)

        dialect = csv.Sniffer().sniff(f.read(1024))
        f.seek(0)

        reader = csv.reader(f, dialect)

        row_nr = 0
        for row in reader:
            if row_nr in range(3):
                pass
            elif row_nr is 4:
                self.set_explicit_axies(row)
            else:
                self.add_data_row(row)
            row_nr += 1

        f.close()

        return SpectralTimetrace(label, self._spectra, self._timepoints,
                                 self._spectral_indices, spectral_unit,
                                 time_unit)


class WavelengthExplicitFile(ExplicitFile):
    """
    Represents a wavelength explicit file
    """
    def get_explicit_axies(self):
        return self._dataset.wavelenghts()

    def set_explicit_axies(self, axies):
        self._spectral_indices = axies

    def get_implicit_axies(self):
        return self.observations()

    def get_data_row(self, index):
        row = []
        for label in self.channel_labels:
            row.append(self.get_channel(label)[index])
        return row

    def add_data_row(self, row):
        if self._timepoints is None:
            self.timepoints = []
        self._timepoints.append(float(row.pop(0)))

        if self._spectra is None:
            self._spectra = []
        self._spectra.append(float(row))

    def get_format_name(self):
        return "Wavelength explicit"


class TimeExplicitFile(ExplicitFile):
    """
    Represents a time explicit file
    """
    def get_explicit_axies(self):
        return self.observations()

    def set_explicit_axies(self, axies):
        self._timepoints = float(axies)

    def get_implicit_axies(self):
        return self.channel_labels

    def get_data_row(self, index):
        return self.get_channel(self.channel_labels()[index])

    def add_data_row(self, row):
        if self._spectral_indices is None:
            self.spectral_indices = []
        self._spectral_indices.append(row.pop(0))

        if self._spectra is None:
            self._spectra = []
        self._spectra.append(float(row))

    def get_format_name(self):
        return "Time explicit"
