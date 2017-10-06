from os.path import join

import numpy as np
from pkg_resources import Requirement, resource_stream
from pytest import fixture

from ..align import match, apply_sum, Alignment
from ..roi_reader import read_roi_zip

TEST_DATA_FOLDER = "plptn/aligner/test/data"


@fixture
def test_roi():
    stream = resource_stream(Requirement.parse('aligner'), join(TEST_DATA_FOLDER, 'test_rois.zip'))
    roi_list = read_roi_zip(stream)
    yield roi_list
    stream.close()


# noinspection PyShadowingNames
@fixture(params=[(0, 'oval_mask_1.npy'), (4, 'oval_mask_2.npy')])
def test_mask(request, test_roi):
    roi_idx, oval_mask = request.param
    mask_stream = resource_stream(Requirement.parse('aligner'), join(TEST_DATA_FOLDER, oval_mask))
    target = np.load(mask_stream)
    yield test_roi[roi_idx], target
    mask_stream.close()


# noinspection PyShadowingNames
def test_oval_mask(test_mask):
    assert np.equal(test_mask[0].mask, test_mask[1]).all()


@fixture
def test_pic():
    from PIL import Image
    stream = resource_stream(Requirement.parse('aligner'), join(TEST_DATA_FOLDER, 'test_pic.jpg'))
    image = Image.open(stream).convert('L')
    yield np.array(image)
    stream.close()


# noinspection PyShadowingNames
def test_match(test_pic):
    target_pic = test_pic[12:-18, 19:-11]
    assert np.equal(match(target_pic, test_pic, np.array([15, 15])), (-4, 3)).all()


# noinspection PyShadowingNames
def test_apply_sum(test_pic):
    target_pic = test_pic[19: -11, 32:-28]
    trans_range = np.array((30, 15))
    summation = np.zeros(np.subtract(test_pic.shape, (30, 60)), dtype=np.uint32)
    apply_sum(summation, test_pic[15: -15, 30: -30], 0, 0)
    apply_sum(summation, target_pic, *-match(target_pic, test_pic, trans_range))
    assert np.equal(summation[10: -10, 10: -10] - test_pic[25: -25, 40: -40].astype(np.uint16) * 2, 0).all()


@fixture
def test_stack():
    from libtiff import TIFF
    stream = resource_stream(Requirement.parse('aligner'), join(TEST_DATA_FOLDER, 'test_stack.tiff'))
    pic = TIFF(stream)
    yield pic
    stream.close()


@fixture
def test_stack_roi():
    stream = resource_stream(Requirement.parse('aligner'), join(TEST_DATA_FOLDER, 'test_stack_roi.zip'))
    roi_list = read_roi_zip(stream)
    yield roi_list
    stream.close()


# noinspection PyShadowingNames
def test_apply_roi(test_stack, test_stack_roi):
    session = Alignment(test_stack, edge_size=(10, 10))
    session.align()
    measurement = session.measure_roi(test_stack_roi)
    for row in measurement['data']:
        assert np.std(row) == 0
