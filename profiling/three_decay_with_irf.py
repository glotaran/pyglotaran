
import numpy as np

from lmfit import Parameters

from glotaran_tools.specification_parser import parse_yml
from glotaran_models.kinetic import KineticSeperableModel

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
x = np.arange(12820, 15120, 4.6)
#  x = np.asarray([0, 1])

wanted_params = Parameters()
wanted_params.add("p1", 101e-3)
wanted_params.add("p2", 202e-4)
wanted_params.add("p3", 505e-5)
wanted_params.add("p4", 0.1)
wanted_params.add("p5", 3.0)
wanted_params.pretty_print()

model = parse_yml(fitspec.format(initial_parameter))

fitmodel = KineticSeperableModel(model)
data = fitmodel.eval(wanted_params, *times, **{'dataset': 'dataset1',
                                               'dataset1_x': x,
                                               })


def fit():
    fitmodel.get_initial_fitting_parameter().pretty_print()
    fitmodel.fit(fitmodel.get_initial_fitting_parameter(),
                 *times, **{"dataset1":
                            data}).best_fit_parameter.pretty_print()


if __name__ == '__main__':
    fit()
