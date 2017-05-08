##
from os.path import expanduser, join

import numpy as np

DATA_FOLDER = "~/Dropbox/script/python/aligner/plptn/aligner/test/data"
##
target = np.zeros((15, 10), dtype=np.bool)
points = np.array([[3, 7], [2, 8], [1, 9], [1, 9], [0, 10], [0, 10], [0, 10], [0, 10]])
points = np.concatenate((points, points[-2:: -1, :]), axis=0)
for idx, row in enumerate(points):
    target[idx, row[0]: row[1]] = True
np.save(join(expanduser(DATA_FOLDER), 'oval_mask_1.npy'), target)
##
target = np.zeros((12, 12), dtype=np.bool)
points = np.array([[4, 8], [2, 10], [1, 11], [1, 11], [0, 12], [0, 12]])
points = np.concatenate((points, points[-1:: -1, :]), axis=0)
for idx, row in enumerate(points):
    target[idx, row[0]: row[1]] = True
np.save(join(expanduser(DATA_FOLDER), 'oval_mask_2.npy'), target)
##
