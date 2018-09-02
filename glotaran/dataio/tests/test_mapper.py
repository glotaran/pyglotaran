import numpy as np
import pytest

from ..mapper import get_pixel_map


@pytest.mark.parametrize("is_array", [True, False])
@pytest.mark.parametrize("shape, transposed, result", [
    ((2, 2, 8), False, ((0, 0), (0, 1), (1, 0), (1, 1))),
    ((2, 2, 8), True, ((0, 0, 1, 1), (0, 1, 0, 1)))
])
def test_get_pixel_map(is_array, shape, transposed, result):
    if is_array:
        in_array = np.arange(np.prod(shape))
        in_array = in_array.reshape(shape)
        pixel_map = get_pixel_map(in_array, transposed=transposed)
        assert pixel_map == result
    else:
        pixel_map = get_pixel_map(shape=shape, transposed=transposed)
        assert pixel_map == result


def test_get_pixel_map_exceptions():
    with pytest.raises(ValueError, match="Using both, `array` and `shape`, is ambiguous. "
                                         "Please only use one of them."):
        in_array = np.array([[[1]]])
        shape = (1, 1, 1)
        get_pixel_map(in_array, shape)

    with pytest.raises(ValueError,
                       match=r"This mapper is designed to map 3 dimensional "
                             r"data \(len\(array.shape\)=3\) to its pixel map \(x-y-plane\). "
                             r"The shape you provided has a dimensionality of 1."):
        in_array = np.array([1])
        get_pixel_map(in_array)

    with pytest.raises(ValueError,
                       match=r"This mapper is designed to map 3 dimensional "
                             r"data \(len\(shape\)=3\) to its pixel map \(x-y-plane\). "
                             r"The shape you provided has a dimensionality of 1."):
        shape = (1,)
        get_pixel_map(shape=shape)

    with pytest.raises(ValueError,
                       match="You to provide either an `array` or a `shape` of dimension 3."):
        get_pixel_map()
