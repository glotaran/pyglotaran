import numpy as np

from glotaran.analysis.fitresult import FitResult

from .irf import IrfGaussian


class KineticFitResult(FitResult):

    def get_sas(self, dataset):
        labels, clp = self.get_clp(dataset)

        dataset = self.model.dataset[dataset].fill(self.model, self.best_fit_parameter)
        compartments = []
        for _, matrix in dataset.get_k_matrices():
            for compartment in matrix.involved_compartments():
                if compartment not in compartments:
                    compartments.append(compartment)
        idx = [labels.index(compartment) for compartment in compartments]
        return compartments, clp[:, idx]

    def get_das(self, dataset, megacomplex):
        labels, clp = self.get_clp(dataset)
        dataset = self.model.dataset[dataset].fill(self.model, self.best_fit_parameter)
        megacomplex = self.model.megacomplex[megacomplex].fill(self.model, self.best_fit_parameter)

        k_matrix = megacomplex.get_k_matrix()

        compartments = dataset.initial_concentration.compartments

        idx = [labels.index(compartment) for compartment in compartments]

        a_matrix = k_matrix.a_matrix(dataset.initial_concentration)

        das = np.dot(clp[:, idx], a_matrix)

        return compartments, das

    def get_coherent_artifact(self, dataset):
        dataset = self.model.dataset[dataset].fill(self.model, self.best_fit_parameter)
        irf = dataset.irf

        if not isinstance(irf, IrfGaussian) or not irf.coherent_artifact:
            return None

        labels = irf.clp_labels()

        time = self.data[dataset.label].get_axis('time')
        spectral = self.data[dataset.label].get_axis('spectral')

        dim1 = len(labels)
        dim2 = spectral.size
        dim3 = time.size

        matrix = np.zeros((dim1, dim2, dim3), dtype=np.float64)

        for i, index in enumerate(spectral):
            _, matrix[:, i, :] = irf.calculate_coherent_artifact(index, time)

        return labels, matrix

    def get_coherent_artifact_clp(self, dataset):
        dataset = self.model.dataset[dataset].fill(self.model, self.best_fit_parameter)
        irf = dataset.irf

        if not isinstance(irf, IrfGaussian) or not irf.coherent_artifact:
            return None

        labels, clp = self.get_clp(dataset)

        idx = [labels.index(label) for label in irf.clp_labels()]

        return irf.clp_labels(), clp[:, idx]
