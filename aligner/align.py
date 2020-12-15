from typing import Tuple, List, Dict, Union
import re
from itertools import islice
from os import path
import os
import json
from pathlib import Path

import cv2
import numpy as np
# from scipy.signal import medfilt2d
from scipy.ndimage import zoom
from noformat import File
from tiffreader import TiffReader, TiffFolderReader, save_tiff, tiffinfo

from .roi_reader import Roi
from .func import masked_mean, apply_sum, apply_frame

EDGE_SIZE = (25, 15)
Json = Dict[str, Union[str, int, List[int]]]
DataFrame = Dict[str, np.ndarray]
Point = Tuple[int, int]


class Alignment(object):
    """img is a temporary pointer to the TIFF file and needs to be created in all functions because otherwise
    it cannot be pickled to be used for multiprocessing."""
    n_frames_for_template = 50
    _template: np.ndarray
    displacement: np.ndarray
    mean_frame: np.ndarray
    std_frame: np.ndarray
    attributes: Json = {}
    two_channels = False
    _fields: Tuple[str, ...] = ('edge_size', 'frame_rate', 'two_channels')
    _channel_no = 1

    def __init__(self, img_path: str, edge_size: Point = EDGE_SIZE) -> None:
        if Path(img_path).is_dir():
            self.img_opener = TiffFolderReader
            self.target_path: str = path.join(path.dirname(img_path), "aligned-" + path.basename(img_path))
        else:
            self.img_opener = TiffReader
            self.target_path: str = path.splitext(img_path)[0]
        self.img_path = img_path
        self.edge_size = edge_size
        self.frame_rate = _get_frame_rate(Path(img_path))

    @staticmethod
    def _get_std(frame: np.ndarray) -> np.ndarray:
        return frame.astype(np.float32).std()
        # return medfilt2d(frame.astype(np.float32)).std()

    @property
    def template(self) -> np.ndarray:
        """create a template from the average frame of a roughly aligned stack"""
        if not hasattr(self, "_template"):
            edge = np.array(self.edge_size, dtype=np.int32)
            img = self.img_opener.open(self.img_path)
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
            self._template = cv2.medianBlur(mean_frame[edge_y1: edge_y2, edge_x1: edge_x2], 3)
        return self._template

    def align(self):
        template, edge = self.template, np.array(self.edge_size, dtype=np.int32)
        img = self.img_opener(self.img_path)
        frame_count, shape = img.length, img.shape
        displacement = list()
        summation = np.zeros(shape, dtype=np.float64)
        sq_summation = np.zeros(shape, dtype=np.float64)
        img.seek(0)
        for frame in iter(img):
            frame = cv2.medianBlur(frame.astype(np.float32), 3)
            translation = match(frame, template, edge)
            displacement.append(translation)
            apply_frame(summation, sq_summation, frame, *translation)
        self.displacement = np.array(displacement)
        self.mean_frame = summation / frame_count
        self.std_frame = np.sqrt(np.maximum((sq_summation - summation ** 2 / frame_count) / frame_count, 0))

    def save(self, target_path: str = None, output_shape: Point = (512, 512),
             draw_limit: bool = False):
        target_path = target_path if target_path else self.target_path
        self.target_path = target_path
        output = File(target_path, 'w+')
        for x in self._fields:
            if hasattr(self, x):
                output.attrs[x] = getattr(self, x)
        output.attrs['img_path'] = path.relpath(self.img_path, target_path)
        if hasattr(self, "displacement"):
            output['displacement'] = self.displacement
        if hasattr(self, "_template"):
            output['template'] = self.template
        for name in ('mean_frame', 'std_frame'):
            if getattr(self, name, None) is not None:
                frame = getattr(self, name)
                if output_shape is not None:
                    frame = zoom(full_contrast(frame), np.divide(output_shape, frame.shape))
                else:
                    frame = full_contrast(frame)
                frame = self.draw_limit(frame) if draw_limit else frame
                save_tiff(frame, path.join(target_path, name + '.tif'))

    @staticmethod
    def _load(cls, file_path: str):
        with open(path.join(file_path, "attributes.json")) as fp:
            attrs = json.load(fp)
        os.chdir(file_path)
        sess = cls(str(Path(attrs["img_path"]).expanduser().resolve()), attrs["edge_size"])
        sess.target_path = file_path
        sess.displacement = np.load(path.join(file_path, "displacement.npy"))
        sess._template = np.load(path.join(file_path, "template.npy"))
        sess.mean_frame = TiffReader.open(path.join(file_path, "mean_frame.tif"))[0]
        sess.std_frame = TiffReader.open(path.join(file_path, "std_frame.tif"))[0]
        return sess

    @classmethod
    def load(cls, file_path: str) -> "Alignment":
        sess = cls._load(cls, file_path)
        if path.exists(path.join(file_path, "measurement.npz")):
            sess.measurement = dict(np.load(path.join(file_path, "measurement.npz")))
        return sess

    def draw_limit(self, frame: np.ndarray, pixel_level: int = 255):
        x1, y1 = self.displacement.max(0)
        x2, y2 = np.flipud(frame.shape) + np.minimum(self.displacement.min(0), 0)
        frame[y1: y2, x1] = pixel_level
        if x2 != frame.shape[1]:
            frame[y1: y2, x2] = pixel_level
        frame[y1, x1: x2] = pixel_level
        if y2 != frame.shape[0]:
            frame[y2, x1: x2] = pixel_level
        return frame

    def measure_roi(self, roi_list: List[Roi]) -> List[DataFrame]:
        tif_path = path.join(self.target_path, 'roi_frame.tif')
        roi_size = tiffinfo(tif_path, ['Image Width: ', 'Image Length: '])  # type: ignore
        original_size = tiffinfo(self.img_path, ['Image Width: ', 'Image Length: '])
        stretch_index = np.divide(original_size, roi_size)
        for roi in roi_list:
            roi.stretch(*stretch_index)
        neuron_names = np.array([_roi_name2int(roi.name) for roi in roi_list])
        time_series = np.divide(np.arange(self.displacement.shape[0]), self.frame_rate)
        result: List[list] = [list() for _ in range(self._channel_no)]
        img = self.img_opener(self.img_path)
        img.seek(0)
        img_iter = iter(img)
        valid_rois = [roi for roi in roi_list if roi.mask.sum() > 10]
        for dx, dy in self.displacement:
            for chan_id in range(self._channel_no):
                frame = cv2.medianBlur(next(img_iter), 3)
                result[chan_id].append([masked_mean(frame, roi.x1 - dx, roi.y1 - dy, roi.mask) for roi in valid_rois])
        return [{'x': neuron_names, 'y': time_series, 'data': np.asarray(x).T} for x in result]

    def save_roi(self, roi_list: List[Roi]) -> Dict[str, np.ndarray]:
        result = self.measure_roi(roi_list)
        save_file = File(self.target_path, 'w+')
        self.measurement = result[0]
        save_file['measurement'] = result[0]
        return self.measurement

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
        if not hasattr(self, "_template"):
            img = self.img_opener(self.img_path)
            edge = np.array(self.edge_size, dtype=np.int32)
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
            self._template = cv2.GaussianBlur(mean_frame[edge_y1: edge_y2, edge_x1: edge_x2], (5, 5), 1)
        return self._template

    def align(self):
        template, edge = self.template, np.array(self.edge_size, dtype=np.int32)
        img = self.img_opener(self.img_path)
        frame_count, shape = img.length // 2, img.shape
        displacement = list()
        summation = np.zeros(shape, dtype=np.float64)
        sq_summation = np.zeros(shape, dtype=np.float64)
        strong_channels = range(self.strong_channel, img.length, 2)
        for frame_id in strong_channels:
            frame = cv2.medianBlur(img[frame_id].astype(np.float32), 3)
            translation = match(frame, template, edge)
            displacement.append(translation)
            apply_frame(summation, sq_summation, frame, *translation)
        self.displacement = np.array(displacement)
        self.mean_frame = summation / frame_count
        self.std_frame = np.sqrt(np.maximum((sq_summation - summation ** 2 / frame_count) / frame_count, 0))

    @classmethod
    def load(cls, file_path: str) -> "AlignmentTwoChannel":
        sess = AlignmentTwoChannel._load(cls, file_path)
        if path.exists(path.join(file_path, "measurement-chan1.npz")):
            sess.measurement = [np.load(path.join(file_path, f"measurement-chan{i}.npz")) for i in range(1, 3)]
        return sess

    def save_roi(self, roi_list: List[Roi]) -> List[Dict[str, np.ndarray]]:  # type: ignore
        result: List[Dict[str, np.ndarray]] = self.measure_roi(roi_list)
        save_file = File(self.target_path, 'w+')
        self.measuremnt = result
        save_file['measurement-chan1'] = result[0]
        save_file['measurement-chan2'] = result[1]
        return result

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

def _get_frame_rate(tif_path: Path) -> float:
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

hz_pattern = re.compile('[-_](\d+)[Hh]z')

def _written_frame_rate(tif_path: Path) -> Union[float, None]:
    match = hz_pattern.search(tif_path.stem)
    return None if match is None else float(match[1])

extra_framerate_pattern = re.compile("frame-rate-((?=.)([+-]?([0-9]*)(\.([0-9]+))?)).info")

def _extra_file_frame_rate(tif_path: Path) -> Union[float, None]:
    if tif_path.is_dir():
        try:
            info_file = next(tif_path.glob("frame-rate-*.info"))
            match = extra_framerate_pattern.match(info_file.name)
            if match:
                return float(match.group(1))
            else:
                raise ValueError(f"frame rate file formatted incorrectly! filename: {info_file.name}")
        except StopIteration:
            return None
    else:
        for file_entry in tif_path.parent.iterdir():
            match = hz_pattern.search(tif_path.stem)
            if match is not None:
                return float(match[1])
    return None

def _tiffinfo_frame_rate(tif_path: Path) -> float:
    import subprocess as sp
    output = sp.check_output(['tiffinfo', '-0', str(tif_path)]).decode('utf-8')
    start_idx = output.find('scanFrameRate')
    try:
        if start_idx == -1:
            start_idx = output.find('state.acq.frameRate')
            return float(output[output.find('=', start_idx) + 1: output.find('\r', start_idx)])
        else:
            return float(output[output.find('=', start_idx) + 2: output.find('\n', start_idx)])
    except ValueError:
        raise ValueError("Fail to find frame rate info inside tiff file.")

def _roi_name2int(roi_name: str) -> int:
    xs = [int(x) for x in roi_name.split('-')]
    return xs[1] + xs[0] * 10000 + sum([x * 100000000 * 10 ** idx for idx, x in enumerate(xs[2:])])

__aligners = {True: AlignmentTwoChannel, False: Alignment}  # switch on attrs['two_channels']

def load_aligner(file_path: str):
    with open(path.join(file_path, "attributes.json")) as fp:
        attrs = json.load(fp)
    cls = __aligners[attrs['two_channels']]
    return cls.load(file_path)
