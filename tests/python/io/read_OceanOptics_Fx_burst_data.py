import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# TODO: read data from zip file
# root_data_path = os.path.join(THIS_DIR, '..', '..', 'resources', 'data', 'ocean_optics_fx_burst.zip')

datapath = r'C:\Users\Joris\Downloads\FX_test_data\test01' # use your path
exportFile = os.path.join(datapath,"test01.ascii")
allFiles = glob.glob(datapath + "/*.txt")
data = []
for index, file_ in enumerate(allFiles):
    print(index)
    # df = pd.read_table(file_, index_col=False, header=None, skiprows=14)
    single_trace = np.genfromtxt(file_, delimiter='\t', skip_header=14)
    if index==0:
        wavelengths = single_trace[:, 0]
    data.append(single_trace[:, 1])
data = np.asarray(data)
times = np.arange(0,data.shape[0])

comments = "# Filename: " + exportFile + "\n" + " \n"
tim = '\t'.join([repr(num) for num in times])
header = comments + "Time explicit\nIntervalnr {}".format(len(times)) + "\n" + tim
raw_data = np.vstack((wavelengths.T, data)).T
np.savetxt(exportFile, raw_data, fmt='%.18e', delimiter='\t', newline='\n', header=header, footer='', comments='')

fig = plt.figure()
plt.xlabel('Wavelength(nm)')
plt.ylabel('Time index')
plt.axis([680, 800, times.min(), times.max()])
plt.pcolormesh(wavelengths, times, data)
plt.show()