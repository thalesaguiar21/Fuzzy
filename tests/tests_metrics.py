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
        for p in [1, 2, 3, 4, 50, 99, 13]:
            try:
                metrics.minkowski(Xtrain, Xtrain, p)
            except Exception as err:
                print(f'Error while computing minkowski with p == {p}')

    def test_zero_dist(self):
        for p in [1, 2, 3, 4, 50, 99, 13]:
            dists = metrics.minkowski(Xtrain, Xtrain, p)
            self.assertTrue(np.all(dists == 0))

    def test_with_single_point(self):
        v = np.array([1, 2, 3])
        s = np.array([4, 2, 3])
        dists = metrics.minkowski(v, s, 2)
        self.assertEqual(3, dists)

    def test_with_1dim(self):
        v = np.array([1])
        s = np.array([3])
        dist = metrics.minkowski(v, s, 2)
        self.assertEqual(2, dist)

    def test_2dim_negative(self):
        v, s = make_negative_2dmin_points()
        for p in [1, 2, 3]:
            dist = metrics.minkowski(v, s, p)
            self.assertGreater(dist, 0)

    def test_with_2darray(self):
        v, s = make_2darrays()
        expected = np.array([[5, 2], [4.1231, 2], [4.0207, 2]])
        for p, d in zip([1, 2, 3], expected):
            dist = metrics.minkowski(v, s, p)
            err = np.abs(dist - d)
            self.assertTrue(np.all(err <= 1e-4))


def make_negative_2dmin_points():
    return np.array([-1, -2]), np.array([-1, -4])


def make_2darrays():
    v = np.array([[1, 3], [2, 4]])
    s = np.array([[-3, 4], [2, 2]])
    return v, s
