from os import listdir, path, chdir, rename, makedirs
from shutil import copy2, copytree
from uifunc import FolderSelector


def distribute_log(folder: str):
    # noinspection PyArgumentList
    files_in_need = [x for x in listdir() if path.splitext(x)[1] in {".tif", ".tiff"} and
                     not path.isfile(path.join(path.splitext(x)[0], 'stimulus.log'))]
    if len(files_in_need) < 1:
        return
    record_times = [(path.getmtime(x), x) for x in files_in_need]
    chdir('stimulus')
    # noinspection PyArgumentList
    log_times = [(float(path.splitext(x)[0]), path.getmtime(x), x) for x in listdir()
                 if path.splitext(x)[1] == '.log']
    chdir(folder)
    for time, file_name in record_times:
        log_file = next(log_name for start, end, log_name in log_times if start < time < end)
        copy2(path.join('stimulus', log_file), path.join(path.splitext(file_name)[0], 'stimulus.log'))


def convert_timestamp(folder: str, ext: str=".log") -> None:
    from datetime import datetime
    for file in listdir(folder):
        if path.splitext(file)[1] != ext:
            continue
        date_time = datetime.strptime(file, "%Y%m%d-%H%M%S{0}".format(ext))
        rename(path.join(folder, file), path.join(folder, str(date_time.timestamp()) + ext))


@FolderSelector
def rearrange_folder(folder: str):
    print("rearranging folder: " + folder)
    date_str = path.split(folder)[1]
    chdir(folder)
    distribute_log(folder)
    for file in listdir(folder):
        base_name, ext = path.splitext(file)
        if ext in {".tiff", ".tif"} and path.isdir(base_name):
            cage_id, animal_id, _, fov_id, *suffix = base_name.split("-")
            fov_folder = path.join('{}-{}'.format(cage_id, animal_id), 'fov-{0}'.format(fov_id))
            makedirs(fov_folder, exist_ok=True)
            copytree(base_name, path.join(fov_folder, date_str + '-'.join([''] + suffix)))
