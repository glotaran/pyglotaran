import numpy as np
from lmfit import Parameters
from glotaran.specification_parser import parse_yml
from glotaran.models.spectral_temporal import model
from _shared import times_with_irf

fitspec = '''
type: kinetic

parameters: {}

compartments: [s1]

megacomplexes:
- label: mc1
  k_matrices: [k1]

k_matrices:
  - label: "k1"
    matrix: {{
      '("s1","s1")': 1
    }}

initial_concentrations: []

irf:
  - label: irf1
    type: gaussian
    center: 4
    width: 5

datasets:
- label: dataset1
  type: spectral
  megacomplexes: [mc1]
  path: ''
  irf: irf1
'''

initial_parameter = [101e-4, 202e-5, 505e-6, 0.1, 4]

times = np.concatenate([np.arange(-10, 1, 0.1).flatten(),
                        np.arange(-1, 1,
                                  0.01).flatten(),
                        np.arange(10, 50, 1.5).flatten(),
                        np.arange(100, 1000,
                                  15).flatten()])
#x = np.arange(12820, 15120, 4.6)
x = np.asarray([0, 1])

wanted_params = Parameters()
wanted_params.add("p_1", 101e-3)
wanted_params.add("p_2", 202e-4)
wanted_params.add("p_3", 505e-5)
wanted_params.add("p_4", 0.1)
wanted_params.add("p_5", 3.0)
wanted_params.pretty_print()

model = parse_yml(fitspec.format(initial_parameter))

model.eval('dataset1',  {"time": times, "spectral": x}, parameter=wanted_params)


def fit():
    result = model.fit()
    result.best_fit_parameter.pretty_print()
    pass

if __name__ == '__main__':
    fit()
    pass

