import numpy as np

from lmfit import Parameters
import time

from glotaran_tools.specification_parser import parse_yml
from glotaran_models.kinetic import KineticSeparableModel

fitspec = '''
type: kinetic

parameter: {}

compartments: [s1, s2, s3]

megacomplexes:
- label: mc1
  k_matrices: [k1]

k_matrices:
  - label: "k1"
    matrix: {{
      '("s1","s1")': 1,
      '("s2","s2")': 2,
      '("s3","s3")': 3,
    }}

initial_concentrations: []

irf:
  - label: irf1
    type: gaussian
    center: 4
    width: 5


dataio:
- label: dataset1
  type: spectral
  megacomplexes: [mc1]
  path: ''
  irf: irf1
'''

initial_parameter = [123e-3, 234e-4, 567e-5, -0.05, 0.1]

times = np.concatenate([np.arange(-10, -1, 0.1).flatten(),
                        np.arange(-1, 10, 0.01).flatten(),
                        np.arange(10, 50, 1.5).flatten(),
                        np.arange(100, 1000, 15).flatten()])
wavenum = np.arange(12820, 15120, 4.6)

simparams = Parameters()
simparams.add("p1", 101e-3)
simparams.add("p2", 202e-4)
simparams.add("p3", 505e-5)
simparams.add("p4", 0.04)
simparams.add("p5", 0.5)
simparams.pretty_print()

model = parse_yml(fitspec.format(initial_parameter))

fitmodel = KineticSeparableModel(model)
start = time.perf_counter()
data = fitmodel.eval(simparams, *times, **{'dataset':'dataset1', 
                                           'noise':True, 'noise_std_dev':0.000001,
                                           'dataset1_x': wavenum,
                                           'amplitudes':[1, 1, 1],
                                          'locations':[14700, 13515, 14180],
                                           'delta': [400,100,300]
                                           })

stop = time.perf_counter()

print("Fitmodel call took: {:.2f}".format(stop - start))


def fit():
    start = time.perf_counter()
    result = fitmodel.fit(fitmodel.get_initial_fitting_parameter(),
                 *times, **{"dataset1":
                            data})
    stop = time.perf_counter()
    result.best_fit_parameter.pretty_print()
    print("Fitting the data took: {:.2f}s".format(stop - start))


if __name__ == '__main__':
    fit()
