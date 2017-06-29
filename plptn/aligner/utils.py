from typing import Union, Tuple, Callable
import numpy as np
from libtiff import libtiff, TIFF
# noinspection PyUnresolvedReferences
from libtiff.libtiff_ctypes import PLANARCONFIG_CONTIG, COMPRESSION_NONE, PLANARCONFIG_SEPARATE


class TiffImage(object):
    _shape = None
    _tiff_ptr = None
    _read_func = None  # type: Callable
    _template_frame = None  # type: np.ndarray

    def __init__(self, tiff_obj: TIFF):
        self._tiff_ptr = tiff_obj

    @classmethod
    def open(cls, filename: Union[str, bytes], mode: str='r'):
        return TiffImage(TIFF.open(filename, mode))

    def __iter__(self):
        yield self.read_current()
        while not libtiff.TIFFLastDirectory(self._tiff_ptr):
            libtiff.TIFFReadDirectory(self._tiff_ptr)
            yield self.read_current()
        raise StopIteration

    def __getitem__(self, item: int) -> np.ndarray:
        libtiff.TIFFSetDirectory(self._tiff_ptr, item)
        return self.read_current()

    def seek(self, index: int):
        libtiff.TIFFSetDirectory(self._tiff_ptr, index)

    @property
    def shape(self) -> Tuple[int, int]:
        if self._shape is None:
            self._shape = (self._tiff_ptr.GetField('ImageLength'), self._tiff_ptr.GetField('ImageWidth'))
        # noinspection PyTypeChecker
        return self._shape

    def read_current(self) -> np.ndarray:
        """ Read image from TIFF and return it as an array. """
        if self._read_func is None or self._template_frame is None:
            get_field = self._tiff_ptr.GetField
            samples_pp = get_field('SamplesPerPixel')  # this number includes extra samples
            samples_pp = 1 if samples_pp is None else samples_pp
            bits = get_field('BitsPerSample')
            sample_format = get_field('SampleFormat')
            planar_config = get_field('PlanarConfig')
            if planar_config is None:  # default is contiguous
                planar_config = PLANARCONFIG_CONTIG
            compression = get_field('Compression')
            compression = None if compression == COMPRESSION_NONE else compression
            self._read_func = libtiff.TIFFReadEncodedStrip if compression else libtiff.TIFFReadRawStrip
            dtype = self._tiff_ptr.get_numpy_type(bits, sample_format)
            if samples_pp == 1:  # only 2 dimensions
                self._template_frame = np.empty(self.shape, dtype)
            elif planar_config == PLANARCONFIG_CONTIG:
                self._template_frame = np.empty((*self.shape, samples_pp), dtype)
            elif planar_config == PLANARCONFIG_SEPARATE:
                self._template_frame = np.empty((samples_pp, *self.shape), dtype)
            else:
                raise IOError("Unexpected PlanarConfig = %d" % planar_config)

        # noinspection PyTypeChecker
        arr = np.empty_like(self._template_frame)
        # actually read strips
        read_func = self._read_func
        pointer = arr.ctypes.data
        size = arr.nbytes
        tiff_ptr = self._tiff_ptr
        for strip in range(libtiff.TIFFNumberOfStrips(tiff_ptr).value):
            # noinspection PyCallingNonCallable
            elem = read_func(tiff_ptr, strip, pointer, max(size, 0)).value
            pointer += elem
            size -= elem
        return arr
