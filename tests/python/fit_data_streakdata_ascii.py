from glotaran.io.wavelength_time_explicit_file import ExplicitFile
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import copy

# Settings:
reproduce_figures_from_paper = False
# Read in streakdata.ascii from resources/data sub-folder
#data_file_te = ExplicitFile('../resources/data/streakdata.ascii')
data_file_te = ExplicitFile('C:\\src\\glotaran\\tests\\resources\\data\\streakdata.ascii')
data_file_te.read("streakdata.ascii")
dataset_te = data_file_te.dataset()
times = dataset_te.get_axis("time")
times = list(np.asarray(times) + 83)
wavelengths = dataset_te.get_axis("spec")

# Get data limits
if reproduce_figures_from_paper:
    [xmin, xmax] = [-20, 200] #with respect to maximum of IRF (needs function written)
    [ymin, ymax] = [630,770]
    linear_range = [-20, 20]
else:
    [xmin,xmax] = [min(dataset_te.get_axis("time")), max(dataset_te.get_axis("time"))]
    [ymin, ymax] = [min(dataset_te.get_axis("spec")),max(dataset_te.get_axis("spec"))]
    linear_range = [-20, 20]
print([xmin,xmax,ymin,ymax])

# Plot the data
axMain = plt.subplot(111)
axMain.pcolormesh(times, wavelengths, dataset_te.data)
axMain.set_xscale('linear')
axMain.spines['right'].set_visible(False)
axMain.yaxis.set_ticks_position('left')
axMain.set_xlim((linear_range[0], linear_range[1]))
axMain.set_ylim(ymin, ymax)
axMain.yaxis.set_ticks_position('left')
axMain.yaxis.set_visible(False)

divider = make_axes_locatable(axMain)
axLog = divider.append_axes("right", size=5.0, pad=0, sharey=axMain)
axLog.set_xscale('log')
axLog.set_xlim((linear_range[1], xmax))
#axLog.xaxis.set_ticks_position('bottom')
#axLog.spines['left'].set_visible(False)
axLog.yaxis.set_visible(False)
#axLog.yaxis.set_ticks_position('right')
axLog.pcolormesh(times, wavelengths, dataset_te.data)
plt.setp(axMain.get_xticklabels(), visible=False)

#ax2 = axLog.twinx()
#ax2.spines['right'].set_visible(False)
#ax2.tick_params(axis='y',which='both',labelright='on')





plt.show()

fitspec = '''
type: kinetic

parameters: {}

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

shapes:
  - label: "shape1"
    type: "gaussian"
    amplitude: shape.amps.1
    location: shape.locs.1
    width: shape.width.1
  - label: "shape2"
    type: "gaussian"
    amplitude: shape.amps.2
    location: shape.locs.2
    width: shape.width.2
  - ["shape3", "gaussian", shape.amps.3, shape.locs.3, shape.width.3]

initial_concentrations: []

irf: []

datasets:
  - label: dataset1
    type: spectral
    megacomplexes: [mc1]
    path: ''
    shapes:
      - compartment: s1
        shape: shape1
      - [s2, shape2]
      - [s3, shape3]

'''

wanted_params = [101e-3, 202e-4, 305e-5]
simparams = copy(wanted_params)
times = np.asarray(np.arange(0, 1500, 1.5))
x = np.arange(12820, 15120, 4.6)
amps = [7, 3, 30, False]
locations = [14700, 13515, 14180, False]
delta = [400, 100, 300, False]

simparams.append({'shape': [{'amps': amps}, {'locs': locations},
                            {'width': delta}]})

model = parse_yml(fitspec.format(simparams))

axies = {"time": times, "spectral": x}

print(model.parameter.as_parameters_dict().pretty_print())

model.eval('dataset1', axies)

print(np.isnan(model.datasets['dataset1'].data.data).any())
print(np.isnan(model.c_matrix()).any())
model.parameter.get("1").value = 300e-3
model.parameter.get("2").value = 500e-4
model.parameter.get("3").value = 700e-5

print(model.parameter.as_parameters_dict().pretty_print())
result = model.fit()
result.best_fit_parameter.pretty_print()
for i in range(len(wanted_params)):
    self.assertEpsilon(wanted_params[i],
                       result.best_fit_parameter["p_{}".format(i + 1)]
                       .value, 1e-6)
