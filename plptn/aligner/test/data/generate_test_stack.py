import numpy as np
from PIL import Image

STACK_NAME = r"test_stack-{0}.tiff"

np.random.seed(12345)
translation = np.random.randn(10, 2).astype(np.int16)


def generate_test_stack():
    temp = np.outer(np.arange(1, 101), np.arange(1, 101)).astype(np.uint16)
    edge = np.array((10, 10), dtype=np.uint16)
    for idx, row in enumerate(translation):
        top_left = edge + row
        bottom_right = row - edge
        Image.fromarray(temp[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0]]).save(STACK_NAME.format(idx))


if __name__ == '__main__':
    generate_test_stack()
