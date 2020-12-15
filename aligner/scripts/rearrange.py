"""rearrange the recording folder to put analysed data accroding to case id and fov id
put stimulus log into the recording folders"""
from typing import Tuple
from os import listdir, path, chdir, rename, makedirs
from datetime import datetime
from shutil import copy2, copytree
from itertools import chain
import numpy as np
from uifunc import FolderSelector


def distribute_log(folder: str):
    # noinspection PyArgumentList
    files_in_need = [x for x in listdir() if path.splitext(x)[1] in {".tif", ".tiff"}
                     and not path.isfile(path.join(path.splitext(x)[0], 'stimulus.log'))]
    if len(files_in_need) < 1:
        return
    record_times = [(path.getmtime(x), x) for x in files_in_need]
    chdir('stimulus')
    # noinspection PyArgumentList
    log_times = [(*get_log_time(x), x) for x in listdir()
                 if path.splitext(x)[1] == '.log']
    chdir(folder)
    for time, file_name in record_times:
        try:
            log_file = next(log_name for start, end, log_name in log_times if start - 0.5 < time < end + 0.5)
        except StopIteration as e:
            print("can't find log: ", file_name, time)
            print("available logs are:")
            for start, end, log_name in log_times:
                print("{0}: start {1}, end {2}".format(log_name, start, end))
            print("recording times are:")
            for time_, file_name_ in record_times:
                print("{0}: end at {1}".format(file_name_, time_))
            raise e
        data_folder = path.splitext(file_name)[0]
        if path.exists(data_folder) and path.isdir(data_folder):
            copy2(path.join('stimulus', log_file), path.join(path.splitext(file_name)[0], 'stimulus.log'))


def convert_timestamp(folder: str, ext: str = ".log") -> None:
    for file in listdir(folder):
        if path.splitext(file)[1] != ext:
            continue
        date_time = datetime.strptime(file, "%Y%m%d-%H%M%S{0}".format(ext))
        rename(path.join(folder, file), path.join(folder, str(date_time.timestamp()) + ext))


def get_log_time(log_path: str) -> Tuple[float, float]:
    end = path.getmtime(log_path)
    start_text = path.splitext(log_path)[0]
    if len(start_text) == 6:
        start_time = datetime.fromtimestamp(end).replace(
            hour=int(start_text[0: 2]), minute=int(start_text[2: 4]), second=int(start_text[4: 6]))
        start = start_time.timestamp()
    else:
        start = float(start_text)
    return start, end


def examine_logs(folder: str) -> np.ndarray:
    stim_folder = path.join(folder, 'stimulus')
    logs = listdir(stim_folder)
    log_mtimes = [path.getmtime(path.join(stim_folder, log)) for log in logs]
    log_starts = [float(path.splitext(log)[0]) for log in logs]
    rec_mtimes = [path.getmtime(path.join(folder, x)) for x in listdir(folder) if path.splitext(x)[1] == '.tif']
    minimum = min(chain(log_mtimes, log_starts, rec_mtimes)) // 1000 * 1000
    rec_times = np.sort(rec_mtimes).reshape(-1, 1) - minimum
    log_times = np.sort(np.vstack([log_starts, log_mtimes]).T, 0) - minimum
    time_iter = zip(rec_times, log_times)
    output = ["log:\t{0[0]:06.0f}\trec:\t{1[0]:06.0f}, {1[1]:06.0f}".format(*next(time_iter))]
    output.extend(["\t{0[0]:06.0f}\t\t{1[0]:06.0f}, {1[1]:06.0f}".format(a, b) for a, b in time_iter])
    if len(rec_times) <= len(log_times):
        output.extend(["\t\t\t\t{0[0]:06.0f}, {0[1]:06.0f}".format(a) for a in log_times[len(rec_times):]])
    else:
        output.extend(["\t{0[0]:06.0f}".format(a) for a in rec_times[len(log_times):]])
    print('\n'.join(output))
    return output


@FolderSelector
def rearrange_folder(folder: str):
    print("rearranging folder: " + folder)
    data_root, date_str = path.split(folder)
    if not date_str:
        data_root, date_str = path.split(folder[0:-1])
    chdir(folder)
    distribute_log(folder)
    for file in listdir(folder):
        base_name, ext = path.splitext(file)
        if ext in {".tiff", ".tif"} and path.isdir(base_name):
            cage_id, animal_id, _, fov_id, *suffix = base_name.split("-")
            fov_folder = path.join(data_root, '{}-{}'.format(cage_id, animal_id), 'fov-{0}'.format(fov_id))
            makedirs(fov_folder, exist_ok=True)
            copytree(base_name, path.join(fov_folder, '-'.join([date_str] + suffix)))
