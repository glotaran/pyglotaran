from glotaran.io.chlorospec_format import ChlorospecData
from glotaran.io.wavelength_time_explicit_file import ExplicitFile
from glotaran.plotting.glotaran_color_codes import get_glotaran_default_colors
import numpy as np
import matplotlib.pyplot as plt

datapath = r'C:\src\glotaran\tests\resources\data\gg\0\0' # use your path
export_path = r'C:\src\glotaran\tests\resources\data\gg.ascii'
data_file_object = ChlorospecData(datapath)

#data_file_object = ChlorospecData('C:\\src\\glotaran\\tests\\resources\\data\\gg\\0\\0')
dataset_te = data_file_object.read("good_game")
times = dataset_te.get_axis("time")
#times = np.linspace(0, 1000, dataset_te.data.shape[1])
plt.plot(dataset_te.get_axis("spectral"),dataset_te.data[:,0])
plt.figure()
plt.contourf(times, dataset_te.get_axis("spectral"), np.nan_to_num(dataset_te.data))
plt.colorbar()
# data_file_object.write('test.ascii',overwrite=True)
exportFile = ExplicitFile(dataset=dataset_te)
exportFile.write(export_path)
plt.show()

