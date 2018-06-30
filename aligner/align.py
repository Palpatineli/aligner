import re
from itertools import islice
from os import path, scandir
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
Json = Dict[str, Union[str, int, List[int]]]
DataFrame = Dict[str, np.ndarray]
Point = Tuple[int, int]


class Alignment(object):
    n_frames_for_template = 50
    _template: np.ndarray = None
    displacement: np.ndarray = None
    mean_frame: np.ndarray = None
    std_frame: np.ndarray = None
    attributes: Json = {}
    two_channels = False
    _fields: Tuple[str, ...] = ('edge_size', 'frame_rate', 'two_channels')
    _channel_no = 1

    def __init__(self, img_path: str, edge_size: Point = EDGE_SIZE) -> None:
        self.img = TiffReader.open(img_path)
        self.img_path = img_path
        self.target_path: str = path.splitext(img_path)[0]
        self.edge_size = edge_size
        self.frame_rate = _get_frame_rate(img_path)

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
            template = (img[best_frame_id][edge[1]: -edge[1], edge[0]: -edge[0]]).astype(np.float32)
            summation = np.zeros(img.shape, dtype=np.float64)
            displacement = list()
            img.seek(0)
            for frame in iter(img):
                frame = frame.astype(np.float32)
                translation = match(frame, template, edge)
                displacement.append(translation)
                apply_sum(summation, frame, *translation)
            mean_translation = np.array(displacement).mean(0).astype(np.int32)
            mean_frame = np.divide(summation, img.length).astype(np.float32)
            edge_x1, edge_y1 = edge + mean_translation
            edge_x2, edge_y2 = mean_translation - edge
            self._template = mean_frame[edge_y1: edge_y2, edge_x1: edge_x2]  # type: np.ndarray
        # noinspection PyTypeChecker
        return self._template

    def align(self):
        template, img, edge = self.template, self.img, np.array(self.edge_size, dtype=np.int32)
        frame_count, shape = img.length, img.shape
        displacement = list()
        summation = np.zeros(shape, dtype=np.float64)
        sq_summation = np.zeros(shape, dtype=np.float64)
        img.seek(0)
        for frame in iter(img):
            frame = frame.astype(np.float32)
            translation = match(frame, template, edge)
            displacement.append(translation)
            apply_frame(summation, sq_summation, frame, *translation)
        self.displacement = np.array(displacement)
        self.mean_frame = summation / frame_count
        self.std_frame = np.sqrt(np.maximum((sq_summation - summation ** 2 / frame_count) / frame_count, 0))

    def save(self, target_path: str = None, output_shape: Point=(512, 512),
             draw_limit: bool=False):
        target_path = target_path if target_path else self.target_path
        output = File(target_path, 'w+')
        for x in self._fields:
            if hasattr(self, x):
                output.attrs[x] = getattr(self, x)
        output.attrs['img_path'] = path.relpath(self.img_path, target_path)
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

    def measure_roi(self, roi_list: List[Roi], stretched: Tuple[int, int]=None) -> List[DataFrame]:
        if stretched is not None:
            stretch_index = np.divide(np.flipud(self.img[0].shape), stretched)
            for roi in roi_list:
                roi.stretch(*stretch_index)
        roi_list = [roi for roi in roi_list if not roi.is_empty]
        neuron_names = np.array([_roi_name2int(roi.name) for roi in roi_list])
        time_series = np.divide(np.arange(self.displacement.shape[0]), self.frame_rate)
        result: List[list] = [list() for _ in range(self._channel_no)]
        self.img.seek(0)
        img_iter = iter(self.img)
        for dx, dy in self.displacement:
            for chan_id in range(self._channel_no):
                frame = next(img_iter)
                result[chan_id].append([masked_mean(frame, roi.x1 - dx, roi.y1 - dy, roi.mask) for roi in roi_list])
        return [{'x': neuron_names, 'y': time_series, 'data': np.asarray(x).T} for x in result]

    def save_roi(self, roi_list: List[Roi], stretched: Point=None, save_path: str=None) -> None:
        result = self.measure_roi(roi_list, stretched)
        save_path = save_path if save_path else self.target_path
        save_file = File(save_path, 'w+')
        save_file['measurement'] = result[0]


class AlignmentTwoChannel(Alignment):
    two_channels = True
    _fields = Alignment._fields + ('strong_channel',)
    _channel_no = 2

    def __init__(self, img_path: str, edge_size: Point = EDGE_SIZE) -> None:
        super(AlignmentTwoChannel, self).__init__(img_path, edge_size)
        self.frame_rate //= 2
        self.strong_channel = 0

    @property
    def template(self) -> np.ndarray:
        if self._template is None:
            img, edge = self.img, np.array(self.edge_size, dtype=np.int32)
            img.seek(0)
            best_frame_id = np.argmax(list(islice(map(self._get_std, iter(img)), self.n_frames_for_template)))
            self.strong_channel = int(best_frame_id % 2)
            template = (img[best_frame_id][edge[1]: -edge[1], edge[0]: -edge[0]]).astype(np.float32)
            summation = np.zeros(img.shape, dtype=np.float64)
            displacement = list()
            for idx in range(self.strong_channel, img.length, 2):
                frame = img[idx].astype(np.float32)
                translation = match(frame, template, edge)
                displacement.append(translation)
                apply_sum(summation, frame, *translation)
            mean_translation = np.array(displacement).mean(0).astype(np.int32)
            mean_frame = np.divide(summation, img.length // 2).astype(np.float32)
            edge_x1, edge_y1 = edge + mean_translation
            edge_x2, edge_y2 = mean_translation - edge
            self._template = mean_frame[edge_y1: edge_y2, edge_x1: edge_x2]  # type: np.ndarray
        # noinspection PyTypeChecker
        return self._template

    def align(self):
        template, img, edge = self.template, self.img, np.array(self.edge_size, dtype=np.int32)
        frame_count, shape = img.length // 2, img.shape
        displacement = list()
        summation = np.zeros(shape, dtype=np.float64)
        sq_summation = np.zeros(shape, dtype=np.float64)
        strong_channels = range(self.strong_channel, img.length, 2)
        for frame_id in strong_channels:
            frame = img[frame_id].astype(np.float32)
            translation = match(frame, template, edge)
            displacement.append(translation)
            apply_frame(summation, sq_summation, frame, *translation)
        self.displacement = np.array(displacement)
        self.mean_frame = summation / frame_count
        self.std_frame = np.sqrt(np.maximum((sq_summation - summation ** 2 / frame_count) / frame_count, 0))

    def save_roi(self, roi_list: List[Roi], stretched: Point=None, save_path: str=None) -> None:
        result = self.measure_roi(roi_list, stretched)
        save_path = save_path if save_path else self.target_path
        save_file = File(save_path, 'w+')
        save_file['measurement-chan1'] = result[0]
        save_file['measurement-chan2'] = result[1]


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

def _tsplit(string, *delimiters):
    pattern = '|'.join(map(re.escape, delimiters))
    return re.split(pattern, string)

def _get_frame_rate(tif_path: str) -> float:
    sample_rate = _written_frame_rate(tif_path)
    if sample_rate:
        return sample_rate
    sample_rate = _extra_file_frame_rate(tif_path)
    if sample_rate:
        return sample_rate
    sample_rate = _tiffinfo_frame_rate(tif_path)
    if sample_rate:
        return sample_rate
    else:
        basename = path.split(tif_path)[-1]
        raise ValueError(f"can't figure out frame_rate for {basename}")

def _written_frame_rate(tif_path: str) -> Union[float, None]:
    basename = path.splitext(path.split(tif_path)[-1])[0]
    if "Hz" in basename:
        return float(next(x for x in _tsplit(basename, '-', '_') if "Hz" in x)[0: -2])
    else:
        return None

def _extra_file_frame_rate(tif_path: str) -> Union[float, None]:
    folder = path.split(tif_path)[0]
    for file_entry in scandir(folder):
        if "Hz" in file_entry.name:
            basename = path.splitext(file_entry.name)[0]
            return float(next(x for x in basename.split("-_") if "Hz" in x)[0: -2])
    return None

def _tiffinfo_frame_rate(tif_path: str) -> float:
    import subprocess as sp
    # noinspection SpellCheckingInspection
    output = sp.check_output(['tiffinfo', '-0', tif_path]).decode('utf-8')
    start_idx = output.find('scanFrameRate')
    if start_idx == -1:
        start_idx = output.find('state.acq.frameRate')
        return float(output[output.find('=', start_idx) + 1: output.find('\r', start_idx)])
    else:
        return float(output[output.find('=', start_idx) + 2: output.find('\n', start_idx)])


def _roi_name2int(roi_name: str) -> int:
    xs = [int(x) for x in roi_name.split('-')]
    return xs[1] + xs[0] * 10000 + sum([x * 100000000 * 10 ** idx for idx, x in enumerate(xs[2:])])


__aligners = {True: AlignmentTwoChannel, False: Alignment}  # switch on attrs['two_channels']


def load_aligner(file_path: str) -> Alignment:
    file = File(file_path, 'r')
    attrs = file.attrs
    trigger = attrs['two_channels']
    cls = __aligners[trigger]
    img_path = attrs['img_path'] + ('' if attrs['img_path'].endswith('.tif') else '.tif')
    if not path.isabs(img_path):
        img_path = path.join(file_path, img_path)
    obj = cls(img_path, attrs['edge_size'])
    obj.target_path = file_path
    for key, value in attrs.items():
        setattr(obj, key, value)
    if 'template' in file:
        obj._template = file['template']
    if 'displacement' in file:
        obj.displacement = file['displacement']
    for frame_type in ('mean_frame', 'std_frame'):
        file_name = frame_type + '.tif'
        for file_entry in scandir(file_path):
            if file_name in file_entry.name:
                setattr(obj, frame_type, TiffReader.open(file_entry.path)[0].astype(np.uint16))
    return obj
