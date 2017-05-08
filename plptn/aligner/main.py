import math
from collections import OrderedDict
from itertools import chain
from os import path, listdir
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from noformat import File
from tiffcapture import opentiff

EDGE_SIZE = 5
SCALE_FACTOR = 0.00390625
TASTE_N_FRAMES = 25
WARP_MAT = np.float32([[1, 0, 0], [0, 1, 0]])


def match(img: np.ndarray, template: np.ndarray) -> Tuple[int, int]:
    res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
    _, _, min_location, _ = cv2.minMaxLoc(res)
    return min_location


def apply_frame(summation: np.ndarray, sq_summation: np.ndarray, frame: np.ndarray, x: int,
                y: int) -> None:
    x_end = x + frame.shape[0]
    y_end = y + frame.shape[1]
    summation[x: x_end, y: y_end] += frame
    sq_summation[x: x_end, y: y_end] += frame.astype(np.uint32) ** 2


def oval_mask(height: int, width: int) -> np.ndarray:
    result = np.zeros((height, width), dtype=np.bool)
    a, b = height / 2 - 0.3, width / 2 - 0.3
    half_height, half_width = (height - 1) / 2, (width - 1) / 2
    for idx, row in enumerate(result):
        span = math.sqrt(1 - ((half_height - idx) / a) ** 2) * b
        row[int(round(half_width - span + 0.5)): int(round(half_width + span + 0.5))] = True
    return result


class Alignment(object):
    _template = None
    displacement = None
    mean_frame = None
    std_frame = None
    target_path = None

    def __init__(self, img_path, edge_size: int = EDGE_SIZE):
        self.img = opentiff(img_path)
        self.edge_size = edge_size
        self.img_path = img_path

    def __del__(self):
        self.img.release()

    @property
    def template(self):
        """create a template from the average frame of a roughly aligned stack"""
        if not self._template:
            img, edge_size = self.img, self.edge_size
            img.seek(0)
            summation = np.zeros(img.shape, dtype=np.uint64)
            std = [np.std(next(img)) for _ in range(TASTE_N_FRAMES)]
            best_frame_id = np.argmax(std) + 1  # type: int
            best_frame = np.multiply(img.find_and_read(best_frame_id),
                                     SCALE_FACTOR).astype(np.uint8)
            template = best_frame[edge_size: -edge_size, edge_size: -edge_size]
            summation += best_frame
            top_left = np.array([edge_size, edge_size])
            for idx in chain(range(best_frame_id), range(best_frame_id + 1, img.length)):
                frame = np.multiply(img.find_and_read(idx), SCALE_FACTOR).astype(np.uint8)
                res = np.subtract(match(frame, template), top_left)
                WARP_MAT[:, 2] = res
                summation += cv2.warpAffine(frame, WARP_MAT, frame.shape)
            mean_frame = np.divide(summation, img.length - 1).astype(np.uint8)
            self._template = mean_frame[edge_size: -edge_size, edge_size: -edge_size]
        return self._template

    def get_displacement(self):
        template, img, edge_size = self.template, self.img, self.edge_size
        length, shape = img.length, img.shape
        top_left = np.array([edge_size, edge_size])
        displacement = np.zeros((length, 2), dtype=np.int32)
        summation = np.zeros(shape, dtype=np.uint64)
        sq_summation = np.zeros_like(shape, dtype=np.uint64)
        for idx in range(length):
            frame = np.multiply(img.find_and_read(idx), SCALE_FACTOR).astype(np.uint8)
            res = np.subtract(match(frame, template), top_left)
            displacement[idx, :] = res
            apply_frame(summation, sq_summation, frame, *(res + top_left))
        self.displacement = displacement
        self.mean_frame = summation / length
        self.std_frame = np.sqrt((sq_summation - summation ** 2 / length) / length)

    @classmethod
    def load(cls, file_path: str):
        file = File(file_path, 'r')
        obj = cls(file.attrs['img_path'], file.attrs['edge_size'])
        obj.target_path = file_path
        if 'template' in file:
            obj._template = file['template']
        if 'displacement' in file:
            obj.displacement = file['displacement']
        file_list = listdir(file_path)
        if 'mean_frame.tif' in file_list:
            obj.mean_frame = np.asarray(Image.open(path.join(file_path, 'mean_frame.tif')),
                                        dtype=np.uint16).T
        if 'std_frame.tif' in file_list:
            obj.std_frame = np.asarray(Image.open(path.join(file_path, 'std_frame.tif')),
                                       dtype=np.uint16).T
        return obj

    def save(self, target_path: str = None):
        if not target_path:
            target_path = self.target_path if self.target_path else self.img_path.rpartition('.')[0]
        output = File(target_path, 'w+')
        output.attrs['img_path'] = self.img_path
        output.attrs['edge_size'] = self.edge_size
        if self.displacement:
            output['displacement'] = self.displacement
        if self._template:
            output['template'] = self.template
        if self.mean_frame:
            temp_img = Image.fromarray(self.mean_frame, 'I')
            temp_img.save(path.join(target_path, 'mean_frame.tif'))
        if self.std_frame:
            temp_img = Image.fromarray(self.std_frame, 'I')
            temp_img.save(path.join(target_path, 'std_frame.tif'))

    def apply_roi(self, roi_list: OrderedDict,
                  stretched: Tuple[int, int]) -> pd.DataFrame:
        column_names = list()
        mask_list = list()
        stretch_index = np.divide(self.img.shape, stretched)
        for key, value in roi_list:
            column_names.extend([key + '_mean', key + '_area'])
            # TODO: convert roi to masks with coordinates
        # TODO: apply these roi mask to displaced frames
        result = np.empty((self.img.length, len(roi_list) * 2))
        return pd.DataFrame(data=result, columns=column_names)
