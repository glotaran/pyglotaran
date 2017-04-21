
import numpy as np

from lmfit import Parameters

from glotaran.specification_parser import parse_yml
from glotaran.model import Dataset
from glotaran.models.kinetic import KineticSeparableModel
import matplotlib.pyplot as plt
import glotaran.plotting.basic_plots as gplt

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

initial_parameter = [0.05, 0, 5]
times = np.asarray(np.arange(-100, 1500, 1.5))
x = np.asarray([1.0, 2.0, 3.0, 4.0])
# x = np.arange(12820, 15120, 46)






def fit():
    return fitmodel.fit(fitmodel.get_initial_fitting_parameter()).best_fit_parameter.pretty_print()

if __name__ == '__main__':
    simulated_params = Parameters()
    simulated_params.add("p1", 0.005)
    simulated_params.add("p2", 0.3)
    simulated_params.add("p3", 10)

    model = parse_yml(fitspec.format(initial_parameter))
    model.eval(simulated_params, 'dataset1', {"time": times, "spec": x})

    fitmodel = KineticSeparableModel(model)
    print(model.datasets['dataset1'].data.data.shape)
    gplt.plot_data_overview(model.datasets['dataset1'].data.get_axis("time"),
                   model.datasets['dataset1'].data.get_axis("spec"),
                   model.datasets['dataset1'].data.data.T)

    fitmodel = fit()

    # TODO: fitmodel.get_residual_matrix()
    plt.show()
