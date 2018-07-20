import matplotlib
import matplotlib.pyplot as plt
from numpy import linalg
from .glotaran_color_codes import get_glotaran_default_colors_cycler, get_glotaran_default_colors
from cycler import cycler

from glotaran.models.spectral_temporal import KineticModel

# TODO: calculate svd when plots are requested
# TODO: calculate svd in background


def plot_data(*args, **kwargs):
    if len(args) == 1 and isinstance(args[0], KineticModel):
        _plot_data_from_kin_sep_model(args[0])
    elif len(args) == 2 and isinstance(args[1], int):
        pass
    elif len(args) == 4 and isinstance(args[0], matplotlib.axes.Axes):
        _plot_data(args[0], args[1], args[2], args[3])


def _plot_data_from_kin_sep_model(model):
    times = model.datasets['dataset1'].data.get_axis("time")
    spectral_indices = model.datasets['dataset1'].data.get_axis("spec")
    data = model.datasets['dataset1'].data.data.T
    plt.pcolormesh(times, spectral_indices, data)


def _plot_data(ax, times, spectral_indices, data):
    ax.pcolormesh(times, spectral_indices, data)


def plot_data_overview(times, spectral_indices, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    ax2.set_prop_cycle(get_glotaran_default_colors_cycler())
    ax3.set_prop_cycle(get_glotaran_default_colors_cycler())
    ax4.set_prop_cycle(cycler('color', get_glotaran_default_colors()))

    xmin = min(times)
    xmax = max(times)
    ymin = min(spectral_indices)
    ymax = max(spectral_indices)

    U1, s1, V1 = linalg.svd(data)
    print("U.shape: {} ; V.shape: {} ; s.shape: {}".format(U1.shape, V1.shape, s1.shape))

    ax1.set_xlim([xmin, xmax])
    ax1.set_ylim([ymin, ymax])
    ax1.set_xlabel('Times (ps)')
    ax1.set_ylabel('$Wavenumber\ [\ cm^{-1}\ ]$')

    plt.tight_layout()
    plot_data(ax1, times, spectral_indices, data)

    ax3.set_xlim([xmin, xmax])
    # ax2.set_ylim([ymin, ymax])
    ax3.set_xlabel('Times (ps)')
    ax3.set_ylabel('$Wavenumber\ [\ cm^{-1}\ ]$')
    plot_trace(ax3, times, V1[0, :].T)

    # ax2.set_ylim([xmin, xmax])
    # ax2.set_ylim([ymin, ymax])
    # ax2.set_xlabel('Times (ps)')
    # ax2.set_ylabel('$Wavenumber\ [\ cm^{-1}\ ]$')
    plot_trace(ax2, U1[:, 0:3], spectral_indices)

    plot_trace(ax4, range(3), s1[0:3])

    # ax1.pcolormesh(times, spectral_indices, data)
    # ax1.set_xlim([xmin, xmax])
    # ax1.set_ylim([ymin, ymax])
    # plot_sing_val_svd

    plt.show(block=False)


def plot_trace(ax, x_values, y_values):
    ax.plot(x_values, y_values)


def plot_residuals():
    pass


def plot_residuals_svd():
    pass


def plot_results():
    pass
