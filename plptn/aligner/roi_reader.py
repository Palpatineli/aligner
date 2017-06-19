from collections import namedtuple
from io import BufferedIOBase, BytesIO
from math import sqrt
from os.path import basename, splitext
from struct import Struct, iter_unpack
from typing import Union, List
from zipfile import ZipExtFile, ZipFile

import numpy as np
from flags import Flags

roi_fmt = Struct('>4sHBx4hH4fH3I2H2BHII')
sub_pixel_fmt = Struct('>4f')
RoiFull = namedtuple("RoiFull", "magic version type y1 x1 y2 x2 n_coordinates fx1 fy1 fx2 "
                                "fy2 stroke_width shape_roi_size stroke_color "
                                "fill_color subtype options style arrow_head_size "
                                "arc_size position header2_offset")
header2_fmt = Struct('>6IHxBIf')
Header2 = namedtuple("Header2", "channel slice frame name_offset name_length overlay_label_color "
                                "overlay_font_size image_opacity image_size float_stroke_width")
# noinspection PyArgumentList
Options = Flags('Options', 'spline_fit double_headed outline overlay_labels overlay_names '
                           'overlay_backgrounds overlay_bold sub_pixel_resolution draw_offset')


# noinspection PyAttributeOutsideInit
class Roi(object):
    __slots__ = ['name', 'y1', 'x1', 'y2', 'x2', 'position']
    __conversion__ = {}
    _mask = None

    # noinspection PyUnusedLocal
    def __init__(self, name: str, base_roi: RoiFull, options: Options, header_2: Header2 = None,
                 extra_data: dict = None):
        self.name = name
        for field_id in self.__slots__:
            target_id = self.__conversion__[field_id] if field_id in self.__conversion__ else field_id
            if hasattr(base_roi, target_id):
                setattr(self, field_id, getattr(base_roi, target_id))
        for field_id in self.__slots__:  # validate self
            if not hasattr(self, field_id):
                raise ValueError("ROI {0} initialization failed. Missing field {1}".format(name, field_id))

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        for field_id in self.__slots__:
            if getattr(self, field_id) != getattr(other, field_id):
                return False
        return True

    @property
    def mask(self) -> np.ndarray:
        raise NotImplementedError

    def stretch(self, x_ratio: float, y_ratio: float) -> None:
        self.x1 = int(round(self.x1 * x_ratio))
        self.x2 = int(round(self.x2 * x_ratio))
        self.y1 = int(round(self.y1 * y_ratio))
        self.y2 = int(round(self.y2 * y_ratio))


class Oval(Roi):
    __type__ = 'oval'

    @property
    def mask(self) -> np.ndarray:
        if self._mask is None:
            width = self.x2 - self.x1
            height = self.y2 - self.y1
            result = np.zeros((height, width), dtype=np.bool)  # type: np.ndarray
            a, b = height / 2 - 0.3, width / 2 - 0.3
            half_height, half_width = (height - 1) / 2, (width - 1) / 2
            for idx, row in enumerate(result):
                span = sqrt(1 - ((half_height - idx) / a) ** 2) * b
                row[int(round(half_width - span + 0.5)): int(round(half_width + span + 0.5))] = True
            self._mask = result
            return result
        else:
            return self._mask


class Rect(Roi):
    __type__ = 'rectangle'
    __slots__ = Roi.__slots__ + ['arc_size']

    @property
    def mask(self) -> np.ndarray:
        return np.ones((self.y2 - self.y1, self.x2 - self.x1), dtype=np.bool)


class Line(Roi):
    __type__ = 'line'
    __conversion__ = dict(Roi.__conversion__, **{'y1': 'fy1', 'x1': 'fx1', 'y2': 'fy2', 'x2': 'fx2'})
    __slots__ = Roi.__slots__ + ['draw_offset']

    def __init__(self, name: str, base_roi: RoiFull, options: Options, header_2: Header2 = None,
                 extra_data: dict = None):
        super(Line, self).__init__(name, base_roi, options, header_2, extra_data)
        self.draw_offset = options.draw_offset

    @property
    def mask(self):
        raise NotImplementedError


class PolyLine(Roi):
    __type__ = 'polyline'
    __slots__ = Roi.__slots__ + ['n_coordinates', 'x_coords', 'y_coords']

    def __init__(self, name: str, base_roi: RoiFull, options: Options, header_2: Header2 = None,
                 extra_data: dict = None):
        super(PolyLine, self).__init__(name, base_roi, options, header_2, extra_data)
        self.x_coords, self.y_coords = extra_data['x_coords'], extra_data['y_coords']

    @property
    def mask(self):
        raise NotImplementedError


class Freehand(PolyLine):
    __type__ = 'freehand'
    __slots__ = PolyLine.__slots__ + ['aspect_ratio', 'ex1', 'ex2', 'ey1', 'ey2']
    __conversion__ = dict(PolyLine.__conversion__,
                          **{'aspect_ratio': 'style', 'ex1': 'fx1', 'ex2': 'fx2', 'ey1': 'fy1', 'ey2': 'fy2'})

    @property
    def mask(self):
        raise NotImplementedError


class Polygon(PolyLine):
    __type__ = 'polygon'

    @property
    def mask(self):
        raise NotImplementedError


class Freeline(PolyLine):
    __type__ = 'freeline'

    @property
    def mask(self):
        raise NotImplementedError


class Traced(PolyLine):
    __type__ = 'traced'

    @property
    def mask(self):
        raise NotImplementedError


class Angle(PolyLine):
    __type__ = 'angle'

    @property
    def mask(self):
        raise NotImplementedError


class Point(PolyLine):
    __type__ = 'point'

    @property
    def mask(self):
        raise NotImplementedError


ROI_TYPE = {0: Polygon, 1: Rect, 2: Oval, 3: Line, 4: Freeline, 5: PolyLine,
            7: Freehand, 8: Traced, 9: Angle, 10: Point}


def read_roi_zip(zip_path: str) -> List[Roi]:
    zf = ZipFile(zip_path)
    return [process_roi_file(zf.open(name)) for name in zf.namelist()]


def process_roi_file(file_path: Union[str, ZipExtFile, BufferedIOBase]) -> Roi:
    """allow different arguments to read_roi_file"""
    if isinstance(file_path, ZipExtFile):
        data = file_path.read()
        name = splitext(basename(file_path.name))[0]
    elif isinstance(file_path, str):
        fp = open(file_path, 'rb')
        data = fp.read()
        fp.close()
        name = splitext(basename(file_path))[0]
    elif isinstance(file_path, BufferedIOBase):
        data = file_path.read()
        name = getattr(file_path, 'name')
        if name:
            name = splitext(basename(name))[0]
    else:
        raise IOError
    return read_roi_file(data, name)


def read_roi_file(data: bytes, name: str) -> Roi:
    stream = BytesIO(data)
    roi_full = RoiFull(*roi_fmt.unpack(stream.read(roi_fmt.size)))
    # noinspection SpellCheckingInspection
    if roi_full.magic != b'Iout':
        raise ValueError("{0}.roi is not a valid ImageJ ROI file".format(name))
    # noinspection PyCallingNonCallable
    options = Options(roi_full.options)
    extra_data = {}
    if options.sub_pixel_resolution and roi_full.version >= 222:
        roi_full = roi_full._replace(y1=roi_full.fy1, x1=roi_full.fx1, y2=roi_full.fy2, x2=roi_full.fx2)
    if roi_full.n_coordinates > 0:
        points = list(iter_unpack('>I', stream.read(roi_full.n_coordinates * 4)))
        x_coords = [roi_full.x1 + x for x in points[0: roi_full.n_coordinates]]
        y_coords = [roi_full.y1 + y for y in points[roi_full.n_coordinates:]]
        if options.sub_pixel_resolution:
            f_points = list(iter_unpack('>f', stream.read(roi_full.n_coordinates * 8)))
            x_coords = [roi_full.fx1 + fx for fx in f_points[0: roi_full.n_coordinates]]
            y_coords = [roi_full.fy1 + fy for fy in f_points[roi_full.n_coordinates:]]
        extra_data.update({'x_coords': x_coords, 'y_coords': y_coords})
    if roi_full.header2_offset > 0 and len(data) > roi_fmt.size + header2_fmt.size + 4:
        stream.seek(roi_full.header2_offset + 4)
        header2 = Header2(*header2_fmt.unpack(stream.read(header2_fmt.size)))
        if any((x > 0 for x in header2[0: 3])):
            roi_full = roi_full._replace(postion=dict(zip(('channel', 'slice', 'frame'), header2[0: 3])))
    else:
        header2 = None
    if roi_full.type in ROI_TYPE:
        roi = ROI_TYPE[roi_full.type](name, roi_full, options, header2, extra_data)
    else:
        raise NotImplementedError("Roi of type number {0} is not implemented".format(roi_full.type))
    return roi
