from ._shared import times_no_irf
from ._shared import times_with_irf
import numpy as np

from lmfit import Parameters

from glotaran.specification_parser import parse_yml
from glotaran.models.kinetic import KineticSeperableModel

fitspec = '''
type: kinetic

parameter: {}

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

irf: []

datasets:
- label: dataset1
  type: spectral
  megacomplexes: [mc1]
  path: ''

'''

init_fit_kinpar_params = [0.03]
times = times_no_irf()

sim_kinpar_params = Parameters()
sim_kinpar_params.add("p1", 0.03)

model = parse_yml(fitspec.format(init_fit_kinpar_params))
wavenum = np.array([680])

fitmodel = KineticSeperableModel(model)
data = fitmodel.eval(sim_kinpar_params, *times, **{'dataset': 'dataset1',
                                                   'dataset1_x': wavenum
                                                   }
                     )


def fit():
    fitmodel.fit(fitmodel.get_initial_fitting_parameter(),
                 *times, **{"dataset1": data})


if __name__ == '__main__':
    fit()
