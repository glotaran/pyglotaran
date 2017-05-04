from glotaran.io.wavelength_time_explicit_file import ExplicitFile
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, SubplotDivider, LocatableAxes, Size
from copy import copy
from glotaran.specification_parser import parse_yml

# Settings:
reproduce_figures_from_paper = True
# Read in streakdata.ascii from resources/data sub-folder
data_file_te = ExplicitFile('../resources/data/streakdata.ascii')
#data_file_te = ExplicitFile('C:\\src\\glotaran\\tests\\resources\\data\\streakdata.ascii')
#data_file_te.read("streakdata.ascii")
dataset_te = data_file_te.read("dataset1")
#dataset_te = data_file_te.dataset()
times = dataset_te.get_axis("time")
times = list(np.asarray(times) + 83)
wavelengths = dataset_te.get_axis("spectral")

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
fig, axMain = plt.subplots(1, 1)
meshLin = axMain.pcolormesh(times, wavelengths, dataset_te.data, cmap='Greys')
axMain.set_xscale('linear')
axMain.spines['right'].set_visible(True)
axMain.yaxis.set_ticks_position('left')
axMain.set_xlim((linear_range[0], linear_range[1]))
axMain.set_ylim(ymin, ymax)
axMain.yaxis.set_ticks_position('left')
axMain.yaxis.set_visible(True)

divider = make_axes_locatable(axMain)
axLog = divider.append_axes("right", size="50%", pad=0, sharey=axMain)
plt.setp(axMain.get_xticklabels(), visible=True)
meshLog = axLog.pcolormesh(times, wavelengths, dataset_te.data, cmap='Greys')
axLog.set_xscale('log')
axLog.set_xlim((linear_range[1], xmax))
axLog.xaxis.set_ticks_position('bottom')
axLog.spines['left'].set_visible(False)
axLog.yaxis.set_visible(False)
axLog.set_ylim(ymin, ymax)
axLog.yaxis.set_ticks_position('right')
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig1 = axLog.get_figure()
fig1.add_axes(ax_cb)
plt.colorbar(meshLog, cax=ax_cb)
#fig.colorbar(meshLin, pad=20.2)

#ax2 = axLog.twinx()
#ax2.spines['right'].set_visible(False)
#ax2.tick_params(axis='y',which='both',labelright='on')

plt.show()

fitspec = '''
type: kinetic

parameters: 
 - -81.0
 - 1.6
 - 0.2
 - 0.06
 - 0.02
 - 0.00016

compartments: [s1, s2, s3, s4]

megacomplexes:
    - label: mc1
      k_matrices: [k1]

k_matrices:
  - label: "k1"
    matrix: {
      '("s1","s1")': 3,
      '("s2","s2")': 4,
      '("s3","s3")': 5,
      '("s4","s4")': 6
    }

irf:
  - label: irf
    type: gaussian
    center: 1
    width: 2

datasets:
  - label: dataset1
    type: spectral
    megacomplexes: [mc1]
    path: ''
    irf: irf

'''

specfit_model = parse_yml(fitspec)
print(specfit_model)
times = np.asarray(dataset_te.get_axis("time"))
wavelengths = np.asarray(dataset_te.get_axis("spectral"))
specfit_model.datasets['dataset1'].data = dataset_te
specfit_result = specfit_model.fit()
specfit_result.best_fit_parameter.pretty_print()
