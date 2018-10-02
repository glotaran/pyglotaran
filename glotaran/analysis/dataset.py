"""This package contains functions for dataset analysis."""

from typing import Tuple
import numpy as np

from glotaran.model.dataset import Dataset


def dataset_svd(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the singular value decomposition of a dataset.

    Parameters
    ----------
    dataset : glotaran.model.dataset.Dataset

    Returns
    -------
    tuple :
        (lsv, svals, rsv)

        lsv : np.ndarray
            left singular values
        svals : np.ndarray
            singular values
        rsv : np.ndarray
            right singular values

    """
    lsv, svals, rsv = np.linalg.svd(dataset.data().T)
    return lsv, svals, rsv.T
