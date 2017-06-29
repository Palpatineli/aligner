from itertools import islice
from os import path, listdir
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd
from libtiff import TIFF, TIFFimage
from noformat import File

from .roi_reader import Oval, Roi

EDGE_SIZE = (50, 15)


class Alignment(object):
    n_frames_for_template = 25
    _template = None
    displacement = None
    mean_frame = None
    std_frame = None
    target_path = None

    def __init__(self, img_path: str, edge_size: Tuple[int, int] = EDGE_SIZE):
        self.img = TIFF.open(img_path, mode='r')
        self.img_path = img_path
        self.edge = np.array(edge_size, dtype=np.int32)

    def __del__(self):
        self.img.close()

    @property
    def template(self) -> np.ndarray:
        """create a template from the average frame of a roughly aligned stack"""
        if self._template is None:
            img, edge = self.img, self.edge
            img.setdirectory(0)
            best_frame_id = np.argmax(list(islice(map(np.std, img.iter_images()), self.n_frames_for_template)))
            template = (img.find_and_read(best_frame_id)[edge[1]: -edge[1], edge[0]: -edge[0]]).astype(np.float32)
            summation = np.zeros(img.shape[::-1], dtype=np.float64)
            img.seek(0)
            for idx, frame in enumerate(img):
                frame = frame.astype(np.float32)
                apply_sum(summation, frame, *match(frame, template, edge))
            mean_frame = np.divide(summation, img.length - 1).astype(np.float32)
            self._template = mean_frame[edge[1]: -edge[1], edge[0]: -edge[0]]  # type: np.ndarray
        # noinspection PyTypeChecker
        return self._template

    def align(self):
        template, img, edge = self.template, self.img, np.array(self.edge)
        length, shape = img.length, img.shape[::-1]
        displacement = np.zeros((length, 2), dtype=np.int32)
        summation = np.zeros(shape, dtype=np.float64)
        sq_summation = np.zeros(shape, dtype=np.float64)
        img.seek(0)
        for idx, frame in enumerate(img):
            frame = frame.astype(np.float32)
            translation = match(frame, template, edge)
            displacement[idx, :] = translation
            apply_frame(summation, sq_summation, frame, *translation)
        self.displacement = displacement - np.round(displacement.mean(0)).astype(np.int32)
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
            obj.mean_frame = TIFF.open(path.join(file_path, 'mean_frame.tif'), mode='r').read_current().astype(np.uint16)
        if 'std_frame.tif' in file_list:
            obj.std_frame = TIFF.open(path.join(file_path, 'std_frame.tif'), mode='r').read_current().astype(np.uint16)
        return obj

    def save(self, target_path: str = None):
        if not target_path:
            target_path = self.target_path if self.target_path else self.img_path.rpartition('.')[0]
        output = File(target_path, 'w+')
        output.attrs['img_path'] = self.img_path
        output.attrs['edge_size'] = self.edge
        if self.displacement is not None:
            output['displacement'] = self.displacement
        if self._template is not None:
            output['template'] = self.template
        if self.mean_frame is not None:
            temp_img = TIFFimage(full_contrast(self.mean_frame), description='average')
            temp_img.write_file(path.join(target_path, 'mean_frame.tif'), compression='none')
        if self.std_frame is not None:
            temp_img = TIFFimage(full_contrast(self.std_frame), description='standard deviation')
            temp_img.write_file(path.join(target_path, 'std_frame.tif'), compression='lzw')

    def measure_roi(self, roi_list: List[Oval],
                    stretched: Tuple[int, int] = None) -> pd.DataFrame:
        if stretched is not None:
            stretch_index = np.divide(np.flipud(self.mean_frame.shape), stretched)
            for roi in roi_list:
                roi.stretch(*stretch_index)
        column_names = [roi.name + '_mean' for roi in roi_list]
        result = np.empty((self.img.length, len(roi_list)))
        self.img.seek(0)
        for idx, frame in enumerate(self.img):
            result[idx, :] = get_measurement(frame, self.displacement[idx], roi_list)
        return pd.DataFrame(data=result, columns=column_names)


def apply_sum(summation: np.ndarray, frame: np.ndarray, x: int, y: int) -> None:
    assert (summation.shape == frame.shape)
    a1 = np.maximum([y, x], 0)
    b1 = np.maximum([-y, -x], 0)
    a2, b2 = frame.shape - b1, frame.shape - a1
    summation[a1[0]: a2[0], a1[1]: a2[1]] += frame[b1[0]: b2[0], b1[1]: b2[1]]


def apply_frame(summation: np.ndarray, sq_summation: np.ndarray, frame: np.ndarray, x: int,
                y: int) -> None:
    a1 = np.maximum([y, x], 0)
    b1 = np.maximum([-y, -x], 0)
    a2, b2 = frame.shape - b1, frame.shape - a1
    summation[a1[0]: a2[0], a1[1]: a2[1]] += frame[b1[0]: b2[0], b1[1]: b2[1]]
    sq_summation[a1[0]: a2[0], a1[1]: a2[1]] += (frame[b1[0]: b2[0], b1[1]: b2[1]]) ** 2


def extract_data(frame: np.ndarray, displacement: Tuple[int, int], roi_list: List[Oval],
                 mask_list: List[np.ndarray]) -> list:
    result = list()
    for roi, mask in zip(roi_list, mask_list):
        patch = frame[roi.y1 - displacement[1]: roi.y2 - displacement[1],
                      roi.x1 - displacement[0]: roi.x2 - displacement[0]]
        result.append(patch[mask].sum())
    return result


def full_contrast(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(x.dtype, np.integer) and np.iinfo(x.dtype).max < (x.max() * 255):
        return np.round((x - x.min()).astype(np.uint32) * 255 / (x.max() - x.min())).astype(np.uint8)
    else:
        return np.round((x - x.min()) * 255 / (x.max() - x.min())).astype(np.uint8)


def match(img: np.ndarray, template: np.ndarray, edge: np.ndarray) -> np.ndarray:
    res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
    _, _, min_location, _ = cv2.minMaxLoc(res)
    if np.any(np.equal(min_location, edge * 2)) or np.any(np.equal(min_location, 0)):
        return np.zeros((2,), dtype=int)
    return np.subtract(edge, min_location)


def get_measurement(frame: np.ndarray, displacement: Tuple[int, int], roi_list: List[Roi]) -> np.ndarray:
    """Return roi measurement for one frame"""
    result = np.zeros((len(roi_list),), dtype=np.uint32)
    for idx, roi in enumerate(roi_list):
        dx, dy = displacement
        result[idx] = frame[roi.y1 - dy: roi.y2 - dy, roi.x1 - dx: roi.x2 - dx][roi.mask].mean()
    return result
