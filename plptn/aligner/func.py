import numpy as np
from numba import jit


@jit(nopython=True, nogil=True, cache=True)
def masked_mean(input_mat, x1, y1, mask):
    y_range, x_range = mask.shape
    count = 0
    sum = 0.0
    for m_y in range(y_range):
        for m_x in range(x_range):
            if mask[m_y, m_x]:
                count += 1
                sum += input_mat[m_y + y1, m_x + x1]
    return sum / count


@jit(nogil=True, nopython=True, cache=True)
def apply_sum(summation: np.ndarray, frame: np.ndarray, x: int, y: int):
    if summation.shape != frame.shape:
        raise ValueError("summation and frame dimensions don't fit")
    a1y, a1x = max(y, 0), max(x, 0)
    b1y, b1x = max(-y, 0), max(-x, 0)
    a2y = frame.shape[0] - b1y
    a2x = frame.shape[1] - b1x
    diff_y = b1y - a1y
    diff_x = b1x - a1x
    for i in range(a1y, a2y):
        for j in range(a1x, a2x):
            summation[i, j] += frame[i + diff_y, j + diff_x]
    return summation


@jit(nogil=True, nopython=True, cache=True)
def apply_frame(summation: np.ndarray, sq_summation: np.ndarray, frame: np.ndarray, x: int, y: int):
    if summation.shape != frame.shape or sq_summation.shape != frame.shape:
        raise ValueError("summation and frame dimensions don't fit")
    a1y, a1x = max(y, 0), max(x, 0)
    b1y, b1x = max(-y, 0), max(-x, 0)
    a2y = frame.shape[0] - b1y
    a2x = frame.shape[1] - b1x
    diff_y = b1y - a1y
    diff_x = b1x - a1x
    for i in range(a1y, a2y):
        for j in range(a1x, a2x):
            temp = frame[i + diff_y, j + diff_x]
            summation[i, j] += temp
            sq_summation[i, j] += temp ** 2
    return summation, sq_summation


