import numpy as np

from glotaran.models.spectral_temporal.kinetic_fit_result import KineticFitResult


class DOASFitResult(KineticFitResult):

    def get_doas(self, dataset):
        labels, clp = self.get_clp(dataset)
        dataset = self.model.dataset[dataset].fill(self.model, self.best_fit_parameter)

        oscillations = []

        for cmplx in dataset.megacomplex:
            for osc in cmplx.oscillations:
                if osc.label not in oscillations:
                    oscillations.append(osc.label)

        dim1 = clp.shape[0]
        dim2 = len(oscillations)
        doas = np.zeros((dim1, dim2), dtype=np.float64)
        for i, osc in enumerate(oscillations):
            sin = clp[:, labels.index(f'{osc}_sin')]
            cos = clp[:, labels.index(f'{osc}_cos')]
            doas[:, i] = np.sqrt(sin*sin+cos*cos)
        return oscillations, doas

    def get_phase(self, dataset):
        labels, clp = self.get_clp(dataset)
        dataset = self.model.dataset[dataset].fill(self.model, self.best_fit_parameter)

        oscillations = []

        for cmplx in dataset.megacomplex:
            for osc in cmplx.oscillations:
                if osc.label not in oscillations:
                    oscillations.append(osc.label)

        dim1 = clp.shape[0]
        dim2 = len(oscillations)
        phase = np.zeros((dim1, dim2), dtype=np.float64)
        for i, osc in enumerate(oscillations):
            sin = clp[:, labels.index(f'{osc}_sin')]
            cos = clp[:, labels.index(f'{osc}_cos')]
            phase[:, i] = np.unwrap(np.arctan2(cos, sin))

        return oscillations, phase
