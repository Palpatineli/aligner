from typing import Tuple
from os import chdir, path, listdir
from glob import iglob
from uifunc import FolderSelector
from ..align import load_aligner
from ..roi_reader import read_roi_zip
from ..utils import tiffinfo


@FolderSelector
def apply_roi(folder_path: str) -> None:
    for sub_folder in listdir(folder_path):
        sub_folder = path.join(folder_path, sub_folder)
        if path.isdir(sub_folder) and path.exists(path.join(sub_folder, 'attributes.json')):
            roi_path = _get_roi_file(sub_folder)
            stretch_size = _get_roi_resolution(roi_path)
            try:
                rois = read_roi_zip(roi_path)
            except IOError as e:
                print('selected alignment has not been ROIed')
                raise e
            if _has_new_roi(sub_folder):
                session = load_aligner(sub_folder)
                session.save_roi(rois, stretch_size, sub_folder)
                print('applied roi for folder ' + sub_folder)


def _get_roi_resolution(roi_path: str) -> Tuple[int, int]:
    try:
        # noinspection PyTypeChecker
        resolutions = path.splitext(roi_path)[0].rpartition('_')[2].split('x')[0:2]
        return (int(resolutions[0]), int(resolutions[1]))
    except (ValueError, IndexError):
        tif_path = path.join(path.split(roi_path)[0], 'roi_frame.tif')
        return tiffinfo(tif_path, ['Image Width: ', 'Image Length: '])  # type: ignore


def _get_roi_file(folder: str) -> str:
    chdir(folder)
    roi_file = next(iglob('*.zip'))
    # noinspection PyTypeChecker
    return path.join(folder, roi_file)


def _has_new_roi(folder: str) -> bool:
    roi_time = path.getmtime(_get_roi_file(folder))
    try:
        measure_file = next(iglob(path.join(folder, '*.zip')))
    except StopIteration:
        return True
    measurement_time = path.getmtime(measure_file)
    return measurement_time <= roi_time
