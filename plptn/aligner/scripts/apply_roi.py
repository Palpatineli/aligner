from typing import List, Tuple
from os import chdir, path
from glob import iglob
from uifunc import FoldersSelector
from ..align import Alignment
from ..roi_reader import read_roi_zip


@FoldersSelector
def apply_roi(folders: List[str]) -> None:
    for folder in folders:
        roi_path, stretch_size = _get_roi_file(folder)
        session = Alignment.load(folder)
        try:
            rois = read_roi_zip(roi_path)
        except IOError as e:
            print('selected alignment has not been ROIed')
            raise e
        session.measure_roi(rois, stretch_size, folder)
        print('applied roi for folder ' + folder)


def _get_roi_file(folder: str) -> Tuple[str, Tuple[int, ...]]:
    chdir(folder)
    roi_file = next(iglob('*.zip'))
    # noinspection PyTypeChecker
    return path.join(folder, roi_file), tuple(map(int, path.splitext(roi_file)[0].rpartition('_')[2].split('x')))
