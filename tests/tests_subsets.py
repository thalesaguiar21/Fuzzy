import unittest

import numpy as np

from .context import fuzzy
from fuzzy.logic import subsets


class TestsGenbell(unittest.TestCase):

    def test_2sub_6mf(self):
        inputs = np.array([[1, 0, 3, 4], [3, 2, 4,1]]).T
        fset = subsets.build_genbell(inputs, 3)
        self.assertEqual(fset.shape, (6, 3))

    def test_zerofeat_space(self):
        inputs = np.array([[0, 0, 0], [0, 0, 0]])
        with self.assertRaises(ValueError):
            fset = subsets.build_genbell(inputs, 3)

    def test_widths(self):
        inputs = np.arange(9).reshape(3, 3)
        fset = subsets.build_genbell(inputs, 3)
        widths = fset[:, 0]
        msg = f'Unexpected feature space {widths} != 6'
        self.assertTrue(all(widths == np.ones(9) * 6), msg)

    def test_platos(self):
        inputs = np.arange(9).reshape(3, 3)
        fset = subsets.build_genbell(inputs, 3)
        platos = fset[:, 1]
        msg = f'Unexpected initial genbell plato: {platos} != 1'
        self.assertTrue(all(platos == 1))

    def test_centers(self):
        inputs = np.arange(9).reshape(3, 3)
        fset = subsets.build_genbell(inputs, 3)
        centers = fset[:, 2]
        expc = [0, 3, 6, 1, 4, 7, 2, 5, 8]
        for i in range(9):
            msg = f'Unexpected initial center at {i}: {centers[i]} != {expc[i]}'
            self.assertEqual(centers[i], expc[i], msg)
    
    def test_centers_non_consecutive(self):
        inputs = np.array([[3, 2], [4, 8], [1, 30]])
        fset = subsets.build_genbell(inputs, 4)
        centers = fset[:, 2]
        expc = [1, 2, 3, 4, 2, 11.333, 20.666, 30]
        for i in range(8):
            msg_ = f'Unexpected initial center at {i}:\n {centers}\n{expc}'
            self.assertAlmostEqual(centers[i], expc[i], 2, msg=msg_)

    def test_centers_with_negative_space(self):
        inputs = np.arange(9).reshape(3, 3) * -1
        fset = subsets.build_genbell(inputs, 3)
        centers = fset[:, 2]
        expc = [-6, -3, 0, -7, -4, -1, -8, -5, -2]
        for i in range(8):
            msg_ = f'Unexpected initial center at {i}:\n {centers}\n{expc}'
            self.assertEqual(centers[i], expc[i], msg=msg_)


