from typing import List
from os import path, chdir, listdir
from noformat import File
from uifunc import FoldersSelector


@FoldersSelector
def abs2rel_paths(folders: List[str]) -> None:
    for folder in folders:
        chdir(folder)
        # noinspection PyArgumentList
        for sub_folder in listdir():
            if path.isdir(sub_folder) and path.exists(path.join(sub_folder, 'attributes.json')):
                data = File(sub_folder, 'w+')
                data.attrs['img_path'] = path.join('..', path.split(data.attrs['img_path'])[1])


if __name__ == '__main__':
    abs2rel_paths()
