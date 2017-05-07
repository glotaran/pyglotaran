import os.path
from os.path import join, dirname, abspath
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    THIS_DIR = dirname(abspath(__file__))
    my_data_path = THIS_DIR # use your path
else:
    # put your own data path here:
    my_data_path = r'C:\src\glotaran\tests\python\io\kevin'

def list_all_csv_files_in_path(path):
    allFiles = glob.glob(path + "/*.csv")
    for index, file_ in enumerate(allFiles):
        print("file_index: {} file_name: {}".format(index, file_))
    return allFiles

def export_to_time_explicit_ascii(export_file_name, data_dict, comments=None):
    if not comments:
        comments = ""
    times = data_dict['times']
    wavelengths = data_dict['wavelengths']
    data = data_dict['data']
    
    header = "# Filename: " + export_file_name + "\n" + comments + " \n"
    tim = '\t'.join([repr(num) for num in times])
    header = header + "Time explicit\nIntervalnr {}".format(len(times)) + "\n" + tim
    raw_data = np.vstack((wavelengths.T, data.T)).T
    np.savetxt(export_file_name, raw_data, fmt='%.18e', delimiter='\t', newline='\n', header=header, footer='', comments='')
    print('exported to {}'.format(export_file_name))

def read_and_clean_data(file,wav_from,wav_to,rename_columns,sanitize=False):
    df = pd.read_csv(file, skipfooter=14, engine='python', header=0, index_col=0)
    # As it turns out, there are some bad values in the time stamps, this line corrects for those:
    df.rename(columns=rename_columns, inplace=True)
    raw_times = np.asarray(df.columns).astype('float64')
    raw_wavelengths = np.asarray(df.index)
    raw_data = df.as_matrix()  # as_matrix returns the appropriate ndarray

    df_subset = df[(df.index > wav_from) & (df.index < wav_to)][:]
    times = np.asarray(df_subset.columns).astype('float64')
    wavelengths = np.asarray(df_subset.index)
    data = df_subset.as_matrix()  # as_matrix returns the appropriate ndarray
    #TODO: warning this is very dangerous, replacing nans and inf with zeros and large values, BEWARE:
    if sanitize:
        data = np.nan_to_num(data)
    return {'data':data,'times':times,'wavelengths':wavelengths,'df':df_subset,'raw_data':raw_data,'raw_times':raw_times,'raw_wavelengths':raw_wavelengths,'raw_df':df}

def plot_data(wavelengths, times, data):
    fig = plt.figure()
    plt.pcolormesh(wavelengths, times, data.T)
    plt.xlabel('Wavelength(nm)')
    plt.ylabel('Time index')
    plt.axis([wavelengths.min(), wavelengths.max(), times.min(), times.max()])
    plt.gca().invert_yaxis()
    plt.show()

# First check if we see the csv files in our data path
all_csv_files = list_all_csv_files_in_path(my_data_path)

file1 = os.path.join(my_data_path,r'PDIC42_150nm_exc490nm_LP2.05mW_vanaf540nm.csv')
export_path_file1 = os.path.join(my_data_path, os.path.splitext(os.path.basename(file1))[0] + '_conv.ascii')

f1 = read_and_clean_data(file1, 540, 900, rename_columns={'0.000000000.1': '0.000000000', '-4.986666667.1': '-4.9866666671', '-4.986666667.2': '-4.98666666712',
                 '-4.986666667.3': '-4.98666666713'})

plot_data(f1['wavelengths'], f1['times'], f1['data'])
export_to_time_explicit_ascii(export_path_file1, f1)

file2 = os.path.join(my_data_path,r'PDIC42150_FS_490exc_PolV_LP2.01mW_UV_tot540nm.csv')
export_path_file2 = os.path.join(my_data_path, os.path.splitext(os.path.basename(file2))[0] + '_conv.ascii')

# I see the data is unusuable below 495
# also beware I filter out NaNs (which might 'destroy' some data, take a look at the raw_data instead)
f2 = read_and_clean_data(file1, 500, 900, rename_columns={'0.000000000.1': '0.000000000', '-4.986666667.1': '-4.9866666671', '-4.986666667.2': '-4.98666666712',
                 '-4.986666667.3': '-4.98666666713'}, sanitize=True)
plot_data(f2['wavelengths'], f2['times'], f2['data'])
export_to_time_explicit_ascii(export_path_file2, f2)