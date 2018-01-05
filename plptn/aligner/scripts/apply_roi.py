from typing import List, Tuple
from os import chdir, path
from glob import iglob
from uifunc import FoldersSelector
from ..align import Alignment
from ..roi_reader import read_roi_zip


@FoldersSelector
def apply_roi(folders: List[str]) -> None:
    for folder in folders:
        roi_path = _get_roi_file(folder)
        stretch_size = _get_roi_resolution(roi_path)
        session = Alignment.load(folder)
        try:
            rois = read_roi_zip(roi_path)
        except IOError as e:
            print('selected alignment has not been ROIed')
            raise e
        session.measure_roi(rois, stretch_size, folder)
        print('applied roi for folder ' + folder)


def _extract(text_str: str, find_str: str) -> int:
    find_str_len = len(find_str)
    start_idx = text_str.find(find_str)
    end_idx = text_str.find(' ', start_idx + find_str_len)
    return int(text_str[start_idx + find_str_len: end_idx])


def _get_roi_resolution(roi_path: str) -> Tuple[int, int]:
    try:
        # noinspection PyTypeChecker
        resolutions = path.splitext(roi_path)[0].rpartition('_')[2].split('x')[0:2]
        return int(resolutions[0]), int(resolutions[1])
    except (ValueError, IndexError):
        import subprocess as sp
        tif_path = path.join(path.split(roi_path)[0], 'roi_frame')
        output = sp.check_output(['tiffinfo', '-0', tif_path]).decode('utf-8')
        width = _extract(output, 'Image Width: ')
        height = _extract(output, 'Image Length: ')
        return width, height


def _get_roi_file(folder: str) -> str:
    chdir(folder)
    roi_file = next(iglob('*.zip'))
    # noinspection PyTypeChecker
    return path.join(folder, roi_file)
