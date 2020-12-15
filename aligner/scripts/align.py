from typing import List
from os import path
import sys

from aligner.align import Alignment, AlignmentTwoChannel
from uifunc import FilesSelector, FoldersSelector

def _run_align(paths, align_class: type):
    for file_path in paths:
        session = align_class(file_path)
        if path.isdir(file_path):
            folder, base = path.split(file_path)
            target = path.join(folder, "align-" + base)
        else:
            folder, base = path.split(path.splitext(file_path)[0])
            target = path.join(folder, base)
        print('aligning ', base)
        session.align()
        session.save(target, draw_limit=True)
        print('alignment finished.')

@FilesSelector(('.tif', '.tiff'))
def align_files(file_paths: List[str]):
    _run_align(file_paths, Alignment)

@FoldersSelector
def align_loose_tifs(folder_paths: List[str]):
    _run_align(folder_paths, Alignment)

@FilesSelector(('.tif', '.tiff'))
def align_two_chan(file_paths: List[str]):
    _run_align(file_paths, AlignmentTwoChannel)
