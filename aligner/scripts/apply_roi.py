from typing import List
from os import chdir, path, listdir
from glob import iglob
from uifunc import FolderSelector

from ..align import load_aligner
from ..roi_reader import read_roi_zip, Roi

@FolderSelector
def apply_roi(folder_path: str) -> None:
    for sub_folder in listdir(folder_path):
        sub_folder = path.join(folder_path, sub_folder)
        if path.isdir(sub_folder) and path.exists(path.join(sub_folder, 'attributes.json')):
            if _has_new_roi(sub_folder):
                rois = get_roi(sub_folder)
                session = load_aligner(sub_folder)
                session.save_roi(rois, sub_folder)
                print('applied roi for folder ' + sub_folder)

def get_roi(sub_folder: str) -> List[Roi]:
    roi_path = _get_roi_file(sub_folder)
    try:
        return read_roi_zip(roi_path)
    except IOError as e:
        print('selected alignment has not been ROIed')
        raise e

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
