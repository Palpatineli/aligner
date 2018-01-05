from itertools import islice
from os import path, listdir
from typing import Tuple, List, Dict, Union

import cv2
import numpy as np
from noformat import File
from scipy.signal import medfilt2d
from scipy.ndimage import zoom

from .roi_reader import Roi
from .utils import TiffReader, save_tiff
from .func import masked_mean, apply_sum, apply_frame

EDGE_SIZE = (25, 15)


class Alignment(object):
    n_frames_for_template = 50
    _template = None        # type: np.ndarray
    displacement = None     # type: np.ndarray
    mean_frame = None       # type: np.ndarray
    std_frame = None        # type: np.ndarray
    target_path = None      # type: str
    attributes = {}         # type: Dict[str, Union[str, int, List[int]]]

    def __init__(self, img_path: str, edge_size: Tuple[int, int] = EDGE_SIZE, two_channels=False):
        self.img = TiffReader.open(img_path)
        self.img_path = img_path
        self.target_path = path.splitext(img_path)[0]
        self.edge_size = edge_size
        self.two_channels = two_channels
        self.frame_rate = _get_frame_rate(img_path) / 2 if two_channels else _get_frame_rate(img_path)
        self.strong_channel = 0

    @staticmethod
    def _get_std(frame: np.ndarray) -> float:
        return medfilt2d(frame.astype(np.float32)).std()

    @property
    def template(self) -> np.ndarray:
        """create a template from the average frame of a roughly aligned stack"""
        if self._template is None:
            img, edge = self.img, np.array(self.edge_size, dtype=np.int32)
            img.seek(0)
            best_frame_id = np.argmax(list(islice(map(self._get_std, iter(img)), self.n_frames_for_template)))
            if self.two_channels:
                self.strong_channel = int(best_frame_id % 2)
            template = (img[best_frame_id][edge[1]: -edge[1], edge[0]: -edge[0]]).astype(np.float32)
            summation = np.zeros(img.shape, dtype=np.float64)
            img.seek(self.strong_channel)  # if two_channels then 0 for chan1 and 1 for chan2, else 0
            displacement = list()
            for idx, frame in enumerate(iter(img)):
                if self.two_channels and idx % 2 == 1:
                    continue
                frame = frame.astype(np.float32)
                translation = match(frame, template, edge)
                displacement.append(translation)
                apply_sum(summation, frame, *translation)
            mean_translation = np.array(displacement).mean(0).astype(np.int32)
            mean_frame = np.divide(summation, img.length // 2 if self.two_channels else img.length).astype(np.float32)
            edge_x1, edge_y1 = edge + mean_translation
            edge_x2, edge_y2 = mean_translation - edge
            self._template = mean_frame[edge_y1: edge_y2, edge_x1: edge_x2]  # type: np.ndarray
        # noinspection PyTypeChecker
        return self._template

    def align(self):
        template, img, edge = self.template, self.img, np.array(self.edge_size, dtype=np.int32)
        length, shape = img.length // 2 if self.two_channels else img.length, img.shape
        displacement = list()
        summation = np.zeros(shape, dtype=np.float64)
        sq_summation = np.zeros(shape, dtype=np.float64)
        img.seek(0)
        for idx, frame in enumerate(iter(img)):
            frame = frame.astype(np.float32)
            translation = match(frame, template, edge)
            displacement.append(translation)
            if not self.two_channels or idx % 2 == self.strong_channel:
                apply_frame(summation, sq_summation, frame, *translation)
        self.displacement = np.array(displacement)
        self.mean_frame = summation / length
        self.std_frame = np.sqrt(np.maximum((sq_summation - summation ** 2 / length) / length, 0))

    @classmethod
    def load(cls, file_path: str):
        file = File(file_path, 'r')
        attrs = file.attrs
        img_path = attrs['img_path']
        if not path.isabs(img_path):
            img_path = path.join(file_path, img_path)
        obj = cls(img_path, attrs['edge_size'], attrs['two_channels'])
        obj.target_path = file_path
        for x in ('frame_rate', 'strong_channel'):
            setattr(obj, x, file.attrs[x])
        if 'template' in file:
            obj._template = file['template']
        if 'displacement' in file:
            obj.displacement = file['displacement']
        for frame_type in ('mean_frame', 'std_frame'):
            file_name = frame_type + '.tif'
            if file_name in listdir(file_path):
                setattr(cls, frame_type, TiffReader.open(path.join(file_path, file_name))[0].astype(np.uint16))
        return obj

    def save(self, target_path: str = None, output_shape: Tuple[int, int]=(512, 512),
             draw_limit: bool=False):
        target_path = target_path if target_path else self.target_path
        output = File(target_path, 'w+')
        fields = ('edge_size', 'frame_rate', 'two_channels', 'strong_channel')
        for x in fields:
            output.attrs[x] = getattr(self, x)
        output.attrs['img_path'] = path.relpath(target_path, self.img_path)
        if self.displacement is not None:
            output['displacement'] = self.displacement
        if self._template is not None:
            output['template'] = self.template
        for name in ('mean_frame', 'std_frame'):
            if getattr(self, name, None) is not None:
                frame = getattr(self, name)
                if output_shape is not None:
                    frame = zoom(full_contrast(frame), np.divide(output_shape, frame.shape))
                frame = self.draw_limit(frame) if draw_limit else frame
                save_tiff(frame, path.join(target_path, name + '.tif'))

    def draw_limit(self, frame: np.ndarray, pixel_level: int=255):
        x1, y1 = self.displacement.max(0)
        x2, y2 = np.flipud(frame.shape) + self.displacement.min(0)
        frame[y1: y2, x1] = pixel_level
        if x2 != frame.shape[1]:
            frame[y1: y2, x2] = pixel_level
        frame[y1, x1: x2] = pixel_level
        if y2 != frame.shape[0]:
            frame[y2, x1: x2] = pixel_level
        return frame

    def measure_roi(self, roi_list: List[Roi], stretched: Tuple[int, int]=None, save_path: str=None,
                    return_result: bool=False) -> Union[Dict[str, np.ndarray],
                                                        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        if stretched is not None:
            stretch_index = np.divide(np.flipud(self.img[0].shape), stretched)
            for roi in roi_list:
                roi.stretch(*stretch_index)
        neuron_names = [roi.name for roi in roi_list]
        self.img.seek(0)
        result = []
        for idx, ((dx, dy), frame) in enumerate(zip(self.displacement, iter(self.img))):
            row = []
            for roi in roi_list:
                row.append(masked_mean(frame, roi.x1 - dx, roi.y1 - dy, roi.mask))
                # row.append(frame[roi.y1 - dy: roi.y2 - dy, roi.x1 - dx: roi.x2 - dx][roi.mask].mean())
            result.append(row)
        result = np.array(result).T
        save_path = save_path if save_path else self.target_path
        save_file = File(save_path, 'w+')
        time_series = np.arange(self.displacement.shape[0]) / self.frame_rate
        if self.two_channels:
            save_file['measurement-chan1'] = {'data': result[:, ::2], 'x': neuron_names, 'y': time_series}
            save_file['measurement-chan2'] = {'data': result[:, 1::2], 'x': neuron_names, 'y': time_series}
            if return_result:
                return save_file['measurement-chan1'], save_file['measurement-chan2']
        else:
            save_file['measurement'] = {'data': result, 'x': neuron_names, 'y': time_series}
            if return_result:
                return save_file['measurement']


def full_contrast(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(x.dtype, np.integer) and np.iinfo(x.dtype).max < (x.max() * 255):
        return np.round((x - x.min()).astype(np.uint32) * 255 / (x.max() - x.min())).astype(np.uint8)
    else:
        return np.round((x - x.min()) * 255 / (x.max() - x.min())).astype(np.uint8)


def match(frame: np.ndarray, template: np.ndarray, edge: np.ndarray) -> np.ndarray:
    min_location = cv2.minMaxLoc(cv2.matchTemplate(frame, template, cv2.TM_SQDIFF_NORMED))[2]
    location = np.subtract(edge, min_location)
    for idx in (0, 1):
        if not -edge[idx] < location[idx] < edge[idx]:
            location[idx] = 0
    return location


def _get_frame_rate(tif_path: str) -> float:
    import subprocess as sp
    # noinspection SpellCheckingInspection
    output = sp.check_output(['tiffinfo', '-0', tif_path]).decode('utf-8')
    start_idx = output.find('scanFrameRate')
    if start_idx == -1:
        start_idx = output.find('state.acq.frameRate')
        return float(output[output.find('=', start_idx) + 1: output.find('\r', start_idx)])
    else:
        return float(output[output.find('=', start_idx) + 2: output.find('\n', start_idx)])
