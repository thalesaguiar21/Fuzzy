import numpy as np
import unittest

from .context import fuzzy
from fuzzy.cluster import metrics

from tests import Xtrain


class TestsMinkowski(unittest.TestCase):

    def test_wrong_p(self):
        for p in [-1, -0.1, 0, 0.1, 0.9999]:
            try:
                metrics.minkowski(Xtrain, Xtrain, p)
                self.fail(f'Computed distance with p == {p}')
            except ValueError:
                pass

    def test_correct_p(self):
        for p in [1, 1.0000001, 1.2, 50, 10, 2, 3]:
            try:
                metrics.minkowski(Xtrain, Xtrain, p)
            except Exception as err:
                print(f'Error while computing minkowski with p == {p}')

    def test_zero_dist(self):
        for p in [1.1, 2, 3, 4, 3.3, 1.9999]:
            dists = metrics.minkowski(Xtrain, Xtrain, p)
            self.assertTrue(np.all(dists == 0))

    def test_with_single_point(self):
        v = [1, 2, 3]
        s = [4, 2, 3]
        dists = metrics.minkowski(v, s, 2)
        self.assertEqual(3, dists)

