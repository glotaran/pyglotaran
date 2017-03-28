import numpy as np

from glotaran.fitmodel import FitModel

from .c_matrix_generator import CMatrixGenerator
from .result import KineticSeparableModelResult


class KineticSeparableModel(FitModel):


    def c_matrix(self, parameter, *args, **kwargs):

        if "dataset" in kwargs:
            label = kwargs["dataset"]
            gen = CMatrixGenerator.for_dataset(self._model, label)
            return gen.calculate(parameter)
        else:
            return self._generator.calculate(parameter)

    def e_matrix(self, **kwargs):
        dataset = self._model.datasets[kwargs['dataset']]
        amplitudes = kwargs["amplitudes"] if "amplitudes" in kwargs else None
        locations = kwargs["locations"] if "locations" in kwargs else None
        delta = kwargs["delta"] if "delta" in kwargs else None
        x = dataset.data.get_axis("spec")
        e = None
        for megacomplex in dataset.megacomplexes:
            cmplx = self._model.megacomplexes[megacomplex]
            k_matrices = [self._model.k_matrices[k] for k in cmplx.k_matrices]
            k_matrix = k_matrices[0] if len(k_matrices) \
                is 1 else k_matrices[0].combine(k_matrices[1:])
            #  E Matrix => channels X compartments
            nr_compartments = len(k_matrix.compartment_map)

            if amplitudes is None:
                tmp = np.full((len(x), nr_compartments), 1.0)
            else:
                tmp = np.empty((len(x), nr_compartments), dtype=np.float64)
                m = k_matrix.compartment_map
                compartments = self._model.compartments
                # translate compartments to indices

                for i in range(len(m)):
                    m[i] = compartments.index(m[i])

                mapped_amps = [amplitudes[i] for i in m]

                for i in range(len(mapped_amps)):
                    for j in range(len(x)):
                        if locations is None or delta is None:
                            tmp[j, i] = mapped_amps[i]
                        else:
                            mapped_locs = [locations[i] for i in m]
                            mapped_delta = [delta[i] for i in m]
                            tmp[:, i] = mapped_amps[i] * np.exp(
                                -np.log(2) * np.square(
                                    2 * (x - mapped_locs[i])/mapped_delta[i]
                                )
                            )

            if e is None:
                e = tmp
            else:
                if e.shape[1] > tmp.shape[1]:
                    for i in range(tmp.shape[1]):
                        e[:, i] = tmp[:, i] + e[:, i]
                else:
                    for i in range(e.shape[1]):
                        tmp[:, i] = tmp[:, i] + e[:, i]
                        e = tmp

            break
        # get the
        return e

    def coefficients(self, *args, **kwargs):
        dataset = self._model.datasets[kwargs['dataset']]

        for megacomplex in dataset.megacomplexes:
            cmplx = self._model.megacomplexes[megacomplex]
            k_matrix = self._get_combined_k_matrix(cmplx)
            m = k_matrix.compartment_map
            compartments = self._model.compartments
            for i in range(len(m)):
                m[i] = compartments.index(m[i])
            e_matrix = self.e_matrix(*args, **kwargs)
            mapped_e_matrix = np.empty(e_matrix.shape, e_matrix.dtype)
            for i in range(len(m)):
                mapped_e_matrix[:, m[i]] = e_matrix[:, i]
            return mapped_e_matrix

    def _parameter_map(self, parameter):
        def map_fun(i):
            if i != 0:
                i = parameter["p{}".format(int(i))]
            return i
        return np.vectorize(map_fun)
