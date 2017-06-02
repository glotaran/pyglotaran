from glotaran.dataio.chlorospec_format import ChlorospecData
from glotaran.dataio.wavelength_time_explicit_file import ExplicitFile
from glotaran.models.spectral_temporal.dataset import SpectralTemporalDataset
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

root_data_path = os.path.join(THIS_DIR, '..', '..', 'resources', 'data', '_chlorospec')
root_export_path = root_data_path
wavelength_calibration_file = os.path.join(root_data_path,'single_spectrum_for_wavelength_calibration.txt')
single_trace = np.genfromtxt(wavelength_calibration_file, delimiter='\t', skip_header=14)
calibrated_wavelengths = single_trace[:, 0]

for exp_name in os.listdir(root_data_path):
    datapath = os.path.join(root_data_path, exp_name)  # use your path
    export_ascii = os.path.join(root_data_path, exp_name +'.ascii')
    export_png = os.path.join(root_data_path, exp_name + '.png')
    if not ChlorospecData.is_valid_path(datapath):
        continue

    data_file_object = ChlorospecData(datapath)
    dataset_raw = data_file_object.read("good_game")
    # using wavelengths from calibration file
    # wavelengths = dataset_raw.get_axis("spectral")
    wavelengths = calibrated_wavelengths
    times = dataset_raw.get_axis("time")
    data = dataset_raw.data

    min_wav_index = np.searchsorted(wavelengths, 600, side='right')
    max_wav_index = np.searchsorted(wavelengths, 800, side='left')
    wavelengths = wavelengths[min_wav_index:max_wav_index]
    data = data[min_wav_index:max_wav_index, :]
    dataset = SpectralTemporalDataset(exp_name)
    dataset.set_axis("time", times)
    dataset.set_axis("spectral", wavelengths)
    dataset.data = data

    #wavelengths = dataset_raw.get_axis("spectral")
    exportFile = ExplicitFile(dataset=dataset)
    exportFile.write(export_ascii, overwrite=True, number_format="%.8e")

    # Specify your range
    linear_range = [0, 1200]
    ymin = 600
    ymax = 800
    xmax = times.max()
    # pick a color (cmap = X):
    # https://matplotlib.org/examples/color/colormaps_reference.html
    my_cmap_color = 'magma'
    # Plot the data
    fig = plt.figure()
    fig.suptitle('Experiment: ' + exp_name, fontsize=14, fontweight='bold')

    axData = fig.add_subplot(111)

    meshLin = axData.pcolormesh(times, wavelengths, dataset.data, cmap=my_cmap_color)
    axData.set_xscale('linear')
    axData.spines['right'].set_visible(True)
    axData.yaxis.set_ticks_position('left')
    axData.set_xlim((linear_range[0], linear_range[1]))
    axData.set_ylim(ymin, ymax)
    axData.yaxis.set_ticks_position('left')
    axData.yaxis.set_visible(True)
    axData.set_xlabel('Time (ms)')
    axData.set_ylabel('Wavelength (nm)')

    divider = make_axes_locatable(axData)
    axLog = divider.append_axes("right", size="50%", pad=0, sharey=axData)
    plt.setp(axData.get_xticklabels(), visible=True)
    meshLog = axLog.pcolormesh(times, wavelengths, dataset.data, cmap=my_cmap_color)
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
    # fig.colorbar(meshLin, pad=20.2)
    plt.suptitle(exp_name)
    plt.savefig(export_png)

    plt.show(block=False)
    # TODO: Build a pdf report:
    # http://matplotlib.org/1.5.3/examples/pylab_examples/multipage_pdf.html
    # pp = PdfPages('test.pdf')
    # plt.savefig(pp, format='pdf')
    # pp.savefig()
    # pp.close()

#block plots
plt.show()