import pickle as pkl
from os.path import join

from pkg_resources import Requirement, resource_stream
from pytest import fixture

from ..roi_reader import read_roi_zip

TEST_DATA_FOLDER = "plptn/reader/test/data"


@fixture
def fov_1_roi():
    stream = resource_stream(Requirement.parse('reader'), join(TEST_DATA_FOLDER, 'fov-1.zip'))
    yield stream
    stream.close()


@fixture
def fov_1_roi_target():
    stream = resource_stream(Requirement.parse('reader'), join(TEST_DATA_FOLDER, 'fov-1-target.pkl'))
    yield pkl.load(stream)
    stream.close()


def test_read_roi_zip(fov_1_roi, fov_1_roi_target):
    roi_list = read_roi_zip(fov_1_roi)
    roi_list_target = fov_1_roi_target
    assert all([x == y for x, y in zip(roi_list_target, roi_list)])
