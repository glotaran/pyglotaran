
import numpy as np

from lmfit import Parameters

from glotaran.specification_parser import parse_yml
from glotaran.model import Dataset
from glotaran.models.kinetic import KineticSeparableModel

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
      '("s1","s1")': 1,
    }}

initial_concentrations: []

irf:
  - label: irf1
    type: gaussian
    center: 2
    width: 3


datasets:
- label: dataset1
  type: spectral
  megacomplexes: [mc1]
  path: ''
  irf: irf1
'''

initial_parameter = [101e-4, 0, 5]
times = np.asarray(np.arange(-100, 1500, 1.5))
# x = np.arange(12820, 15120, 4.6)
x = np.asarray([1.0, 2.0])

#  axies = IndependentAxies()
#  axies.add(x)
#  axies.add(times)

wanted_params = Parameters()
wanted_params.add("p1", 101e-3)
wanted_params.add("p2", 0.3)
wanted_params.add("p3", 10)

model = parse_yml(fitspec.format(initial_parameter))

fitmodel = KineticSeparableModel(model)
model.eval(wanted_params, 'dataset1', {"time": times, "spec": x})


# print(model.datasets['dataset1'].data.data.shape)


def fit():
    fitmodel.fit(fitmodel.get_initial_fitting_parameter()).best_fit_parameter.pretty_print()
    pass

if __name__ == '__main__':
    fit()
    pass
