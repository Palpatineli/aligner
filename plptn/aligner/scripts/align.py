from typing import List
from os import path

from plptn.aligner.align import Alignment
from uifunc import FilesSelector


@FilesSelector(('.tif', '.tiff'))
def align_files(file_paths: List[str]):
    for file_path in file_paths:
        session = Alignment(file_path)
        print('aligning ', path.splitext(file_path)[0])
        session.align()
        session.save(path.splitext(file_path)[0], draw_limit=True)
        print('alignment finished.')


@FilesSelector(('.tif', '.tiff'))
def align_two_chan(file_paths: List[str]):
    for file_path in file_paths:
        session = Alignment(file_path, two_channels=True)
        print('aligning ', path.splitext(file_path)[0])
        session.align()
        session.save(path.splitext(file_path)[0], draw_limit=True)
        print('alignment finished.')
