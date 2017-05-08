import unittest
from os.path import join

import numpy as np
from pkg_resources import Requirement, resource_stream

from plptn.aligner.main import oval_mask

TEST_DATA_FOLDER = "plptn/aligner/test/data"


class TestAlignerUtil(unittest.TestCase):
    def test_oval_mask(self):
        target = np.load(resource_stream(Requirement.parse('aligner'), join(TEST_DATA_FOLDER, 'oval_mask_1.npy')))
        assert ((oval_mask(15, 10) == target).all())
        target = np.load(resource_stream(Requirement.parse('aligner'), join(TEST_DATA_FOLDER, 'oval_mask_2.npy')))
        assert ((oval_mask(12, 12) == target).all())


if __name__ == '__main__':
    unittest.main()
