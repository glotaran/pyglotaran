"""This package contains functions for dataset analysis."""

from typing import Tuple
import numpy as np

from ..datasets.dataset import Dataset


def dataset_svd(dataset: Dataset) -> Tuple[np.array, np.array, np.array]:
    """
    Returns the singular value decomposition of a dataset.

    Parameters
    ----------
    dataset : glotaran.model.dataset.Dataset

    Returns
    -------
    tuple :
        (lsv, svals, rsv)

        lsv : np.array
            left singular values
        svals : np.array
            singular values
        rsv : np.array
            right singular values

    """
    lsv, svals, rsv = np.linalg.svd(dataset.data().T)
    return lsv, svals, rsv.T
