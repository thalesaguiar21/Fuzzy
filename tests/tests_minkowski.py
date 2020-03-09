import unittest
import math
import os

import numpy as np

from .context import fuzzy
from fuzzy.cluster.fknn import minkowski

class TestsMinkowski(unittest.TestCase):

    def setUp(self):
        self.p1 = np.array([2, 4])
        self.p2 = np.array([3, 2])

    def test_euclidean(self):
        squared_diff = (self.p1 - self.p2)**2
        edist = math.sqrt(squared_diff.sum())
        self.assertAlmostEqual(edist, minkowski(self.p1, self.p2, 2))

    def test_manhattan(self):
        diff = self.p1 - self.p2
        dist = np.abs(diff).sum()
        self.assertEqual(dist, minkowski(self.p1, self.p2, 1))

    def test_negatives(self):
        self.p1 = np.array([-1, 3])
        self.p2 = np.array([0, -5])
        squared_diff = (self.p1 - self.p2)**2
        edist = math.sqrt(squared_diff.sum())
        self.assertAlmostEqual(edist, minkowski(self.p1, self.p2, 2))

    def test_with_list(self):
        try:
            minkowski([1, 1], [1, 1], 2)
        except:
            self.fail()

    def test_diff_dim(self):
        try:
            minkowski([1, 2, 3], [2, 1], 1)
            self.fail()
        except ValueError:
            pass

