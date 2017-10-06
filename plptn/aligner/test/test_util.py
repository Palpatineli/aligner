from pytest import fixture
import numpy as np
from ..utils import bisect


@fixture
def sequence():
    result = np.full(10000, 1, dtype=np.uint8)
    result[0:2573] = 5
    return result


# noinspection PyShadowingNames
def test_bisect(sequence):
    assert bisect(lambda x: sequence[x] > 3, 0, len(sequence)) == 2573
    assert bisect(lambda x: sequence[x] > 3, 2575, len(sequence)) == 2575
    assert bisect(lambda x: sequence[x] > 3, 0, 1000) == 1000
