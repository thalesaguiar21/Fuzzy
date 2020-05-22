import unittest

import numpy as np

from .context import fuzzy
from fuzzy.logic import subsets


class TestsSubsets(unittest.TestCase):

    def test_2sub_6mf(self):
        inputs = np.array([[1, 0, 3, 4], [3, 2, 4,1]]).T
        fset = subsets.build_genmf(inputs, 3)
        self.assertEqual(fset.shape, (6, 3))

    def test_zerofeat_space(self):
        inputs = np.array([[0, 0, 0], [0, 0, 0]])
        with self.assertRaises(ValueError):
            fset = subsets.build_genmf(inputs, 3)

    def test_widths(self):
        inputs = np.arange(9).reshape(3, 3)
        fset = subsets.build_genmf(inputs, 3)
        widths = fset[:, 0]
        msg = f'Unexpected feature space {widths} != 6'
        self.assertTrue(all(widths == np.ones(9) * 6), msg)

