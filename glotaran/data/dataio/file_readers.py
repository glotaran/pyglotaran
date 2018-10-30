"""
Glotarans module to read files
"""
from typing import Callable, Dict, List, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from glotaran.data.external_file_readers.sdt_reader import SdtFile
from .mapper import get_pixel_map
from glotaran.data.datasets.spectral_temporal_dataset import SpectralTemporalDataset
from glotaran.data.datasets.specialized_datasets import FLIMDataset
from glotaran.model.dataset import DimensionalityError


def DataFrame_to_SpectralTemporalDataset(input_dataframe: pd.DataFrame,
                                         time_unit: str = "s",
                                         spectral_unit: str = "nm",
                                         swap_axis: bool = False) \
        -> SpectralTemporalDataset:
    """
    Uses a pd.DataFrame to generate a SpectralTemporalDataset from it.
    The basic assumption is that the pd.DataFrame is time explicit (index=wavelength axis,
    columns=time axis).
    If that isn't the case, the `swap_axis` can be used to transform
    a wavelength explicit pd.DataFrame to a time explicit one internally, so the proper
    SpectralTemporalDataset will be returned.

    Parameters
    ----------
    input_dataframe: pd.DataFrame
        DataFrame containing the data and axes used to generate the SpectralTemporalDataset.
        If the DataFrame isn't time explicit (index=wavelength axis, columns=time axis),
        but wavelength explicit use `swap_axis=True` to get the proper SpectralTemporalDataset.

    time_unit: str: {'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs'}, default 's'
        Unit the time axis values were saved in.

    spectral_unit: str: {'um', 'nm', 'cm^-1'}, default 'nm'
        Unit the wavelength axis values were saved in.

    swap_axis: bool, default False
        Flag to switch a wavelength explicit `input_df` to time explicit `input_df`,
        before generating the SpectralTemporalDataset.

    Raises
    ______
    ValueError:
        If input_df.columns isn't convertible to numeric values.
    ValueError:
        If input_df.index isn't convertible to numeric values.

    Returns
    -------
    SpectralTemporalDataset
        Dataset containing the values and axis from input_df.

    See Also
    --------
    sdt_to_df
    read_sdt
    """
    if swap_axis:
        input_dataframe = input_dataframe.T
    STDataset = SpectralTemporalDataset(time_unit, spectral_unit)
    try:
        time_axis = pd.to_numeric(np.array(input_dataframe.columns))
    except ValueError:
        time_axis = "columns" if not swap_axis else "index"
        raise ValueError(f"The {time_axis} of the DataFrame needs to be convertible "
                         f"to numeric values.")
    # there might be no need for wavelengths to be numeric
    # but ensuring it will prevent problems with string escaping if the
    # dataframe gets saved
    try:
        spectral_axis = pd.to_numeric(np.array(input_dataframe.index))
    except ValueError:
        spectral_axis = "index" if not swap_axis else "columns"
        raise ValueError(f"The {spectral_axis} of the DataFrame needs to be convertible "
                         f"to numeric values.")
    STDataset.time_axis = time_axis
    STDataset.spectral_axis = spectral_axis
    STDataset.data = input_dataframe.values
    return STDataset


def DataFrame_to_FLIMDataset(input_dataframe: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                             mapper_function: Callable[[np.ndarray],
                                                       Union[List[tuple],
                                                             Tuple[tuple],
                                                             np.ndarray]],
                             orig_shape: tuple, orig_time_axis_index: int = 2,
                             time_unit: str = "s", swap_axis: bool = False) \
        -> FLIMDataset:
    """
    Uses a pd.DataFrame to generate a FLIMDataset from it.
    The basic assumption is that the pd.DataFrame is time explicit (index=pixel axis,
    columns=time axis).
    If that isn't the case, the `swap_axis` can be used to transform
    a pixel explicit pd.DataFrame to a time explicit one internally, so the proper
    FLIMDataset will be returned.

    Parameters
    ----------
    input_dataframe: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        DataFrame containing the data and axes used to generate the FLIMDataset.

        Or dict with the keys 'time_traces' and 'intensity_map',
        see: `FLIM_legacy_to_df`

        If the DataFrame isn't time explicit (index=pixel axis, columns=time axis),
        but pixel explicit use `swap_axis=True` to get the proper FLIMDataset.

    mapper_function: Callable[[np.ndarray], Union[List[tuple], Tuple[tuple], np.ndarray]]

        Mapper function/method which is used to to reduce the
        dimensionality of the high dimensional data.

    orig_shape: tuple
        Original shape of the high dimensional data.

    orig_time_axis_index: int
        Index of the axis which corresponds to the time axis.
        I.e. for data of shape (64, 64, 256), which are a 64x64 pixel map
        with 256 time steps, orig_time_axis_index=2.

    time_unit: str: {'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs'}, default 's'
        Unit the time axis values were saved in.

    swap_axis: bool, default False
        Flag to switch a pixel explicit `input_df` to time explicit `input_df`,
        before generating the FLIMDataset.

    Raises
    ______
    ValueError:
        If input_df.columns isn't convertible to numeric values.

    Returns
    -------
    FLIMDataset:
        Dataset containing the values and axis from input_df.

    See Also
    --------
    sdt_to_df
    read_sdt
    FLIM_legacy_to_df
    """
    is_legacy = False
    if isinstance(input_dataframe, dict) and list(input_dataframe.keys()) == ['time_traces',
                                                                              'intensity_map']:
        is_legacy = True
        intensity_map = input_dataframe['intensity_map'].values
        input_dataframe = input_dataframe['time_traces']
    if swap_axis:
        input_dataframe = input_dataframe.T
    flim_dataset = FLIMDataset(mapper_function, orig_shape, time_unit)
    try:
        time_axis = pd.to_numeric(np.array(input_dataframe.columns))
    except ValueError:
        time_axis = "columns" if not swap_axis else "index"
        raise ValueError(f"The {time_axis} of the DataFrame needs to be convertible "
                         f"to numeric values.")
    flim_dataset.time_axis = time_axis
    flim_dataset.pixel_axis = np.array(input_dataframe.index)
    flim_dataset.data = input_dataframe.values
    if not is_legacy:
        intensity_map = input_dataframe.values.reshape(orig_shape).sum(axis=orig_time_axis_index)
    flim_dataset.intensity_map = intensity_map
    return flim_dataset


def sdt_to_DataFrame(file_path: str, index: Union[list, np.ndarray, tuple] = None,
                     dataset_index: int = None,
                     mapper_function: Callable[[np.ndarray], Tuple[tuple]] = None) \
        -> Tuple[pd.DataFrame, tuple]:
    """
    Reads and `*.sdt` file and returns a pd.DataFrame.
    Due to the different nature of the data save in a `*.sdt` files,
    it might be needed to provide an `index` or a `mapper_function`.

    Parameters
    ----------
    file_path: str
        filename or absolute path to the file which should be read

    index: Union[list, np.ndarray, tuple]
        Index to be used for the pd.DataFrame, in case it wasn't
        provided by the `*.sdt` file or the mapper.

    dataset_index: Union[int, None], default None
        This is only needed if the given `*.sdt` file contains multiple
        Datasets.

    mapper_function: Callable, default None
        Function used to generate an index for none flat data,
        which than is used as the index for the flattened data.

        Flat data are data than can be represented as a table (2D), i.e. a
        SpectralTemporalDataset of dimension (M x N).

        None flat data are of higher dimension (>=3D), i.e. FLIM data of form
        (M x M x N).

    Raises
    ______
    ValueError:
        If mapper_function doesn't preserve one dimension, mainly time axis.
    DimensionalityError:
        If mapper_function did produce none flat data
    IndexError:
        If index doesn't have the dimension of data.shape[0]

    Returns
    -------
    pd.DataFrame
        Time explicit DataFrame (columns of the DataFrame is the time axis)

    See Also
    --------
    read_sdt

    """
    sdt_parser = SdtFile(file_path)
    if not dataset_index:
        # looking at the source code of SdtFile, times and data
        # always have the same len, so only one needs to be checked
        nr_of_datasets = len(sdt_parser.times)
        if nr_of_datasets > 1:
            warnings.warn(UserWarning(
                f"The file '{file_path}' contains {nr_of_datasets} Datasets.\n "
                f"By default only the first Dataset will be read. "
                f"If you only need the first Dataset and want get rid of "
                f"this warning you can set dataset_index=0.")
            )
        dataset_index = 0
    times = sdt_parser.times[dataset_index]
    data = sdt_parser.data[dataset_index]
    orig_shape = data.shape
    if mapper_function:
        index = mapper_function(array=data)
        target_shape = (len(index), int(np.prod(data.shape)/len(index)))
        data = data.reshape(target_shape)
        if data.shape[1] not in orig_shape:
            raise ValueError(f"The provided mapper wasn't sufficient, since the "
                             f"shape of the data is {data.shape} and one value of the original "
                             f"shape {orig_shape} needs to be preserved.")
    elif not mapper_function and len(orig_shape) > 2:
        raise DimensionalityError(f"The data you try to read are of shape {orig_shape}, "
                                  f"those data need to be flattened, which is done by "
                                  f"utilizing a mapper function. The mapper function should "
                                  f"provide the indices for the flattened data.")

    elif index and len(index) is not data.shape[0]:
        raise IndexError(f"The Dataset contains {data.shape[0]} measurements, but the "
                         f"indices supplied are {len(index)}.")
    elif not index:
        warnings.warn(UserWarning(
            f"There was no `index` provided."
            f"That for the indices will be a entry count(integers)."
            f"To prevent this warning from being shown, provide "
            f"a list of indices, with len(index)={data.shape[0]}")
            )
    return pd.DataFrame(data, index=index, columns=times), orig_shape


def read_sdt(file_path: str, index: Union[list, np.ndarray] = None,
             type_of_data: str = "st", time_unit: str = "s", swap_axis: bool = False,
             dataset_index: int = None, return_dataframe: bool = False,
             orig_time_axis_index: int = 2, spectral_unit: str = "nm") \
        -> Union[pd.DataFrame, SpectralTemporalDataset, FLIMDataset]:
    """
    Reads a `*.sdt` file and returns a pd.DataFrame (`return_dataframe==True`), a
    SpectralTemporalDataset (`type_of_data=='st'`) or a FLIMDataset (`type_of_data=='flim'`).

    Parameters
    ----------
    file_path: str
        Path to the sdt file which should be read.

    index: list, np.ndarray
        This is only needed if `type_of_data=="st"`, since `*.sdt` files,
        which only contain spectral temporal data, lack the spectral information.
        Thus for the spectral axis data need to be given by the user.

    type_of_data: str: {'st', 'flim'}, default 'st'
        As long as `return_dataframe==False` the following types will be returned.

        'st':
            A SpectralTemporalDataset will be returned
        'flim':
            A FLIMDataset will be returned

    time_unit: str: {'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs'}, default 's'
        Time unit the data was saved in.

    swap_axis: bool, default False
        Flag to switch none to time explicit data to time explicit data,
        before generating the Dataset.

    dataset_index: int: default 0
        If the `*.sdt` file contains multiple datasets the index will used
        to select the wanted one

    return_dataframe: bool: default False
        This flag can be used to force the reader to return a pandas.DataFrame object.
        With this Dataframe object it can be quickly determined if the reading of the file
        gave the proper result

    orig_time_axis_index: int
        Index of the axis which corresponds to the time axis.
        I.e. for data of shape (64, 64, 256), which are a 64x64 pixel map
        with 256 time steps, orig_time_axis_index=2.

    spectral_unit: str: {'um', 'nm', 'cm^-1'}, default 'nm'
        This is only needed if `type_of_data=="st"`

    Raises
    ______
    ValueError:
        If `type_of_data` isn't in ["st", "flim"]

    Returns
    -------
    Union[pd.DataFrame, SpectralTemporalDataset, FLIMDataset]:
        pd.DataFrame:
            If `return_dataframe==True`
        SpectralTemporalDataset:
            If `type_of_data=='st'` and `return_dataframe==False`
        FLIMDataset:
            If `type_of_data=='flim'` and `return_dataframe==False`

    See Also
    --------
    df_to_SpectralTemporalDataset
    df_to_FLIMDataset
    sdt_to_df

    """
    supported_type_of_data = ["st", "flim"]
    if type_of_data not in supported_type_of_data:
        raise ValueError(f"The entered value of `type_of_data` was {repr(type_of_data)}, "
                         f"this value isn't supported. The supported values are "
                         f"{repr(supported_type_of_data)}.")
    if type_of_data == "flim":
        data_dataframe, orig_shape = sdt_to_DataFrame(file_path=file_path,
                                                      dataset_index=dataset_index,
                                                      mapper_function=get_pixel_map)
    else:
        data_dataframe, orig_shape = sdt_to_DataFrame(file_path=file_path, index=index,
                                                      dataset_index=dataset_index)
    if return_dataframe:
        return data_dataframe
    else:
        if type_of_data == "flim":
            dataset = DataFrame_to_FLIMDataset(input_dataframe=data_dataframe,
                                               mapper_function=get_pixel_map,
                                               orig_shape=orig_shape,
                                               orig_time_axis_index=orig_time_axis_index,
                                               time_unit=time_unit,
                                               swap_axis=swap_axis)
        else:
            dataset = DataFrame_to_SpectralTemporalDataset(input_dataframe=data_dataframe,
                                                           time_unit=time_unit,
                                                           spectral_unit=spectral_unit,
                                                           swap_axis=swap_axis)
        return dataset
