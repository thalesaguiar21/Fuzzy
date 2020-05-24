import unittest

import numpy as np

from .context import fuzzy
from fuzzy.cluster import kdtree


DATA = [[0, 1, 0], [4, 3, 1], [2, 1, 0],
        [-3, 2, 0], [15, -8, 1], [2, -2, 0]]


class TestsKdTree(unittest.TestCase):

    def test_root(self):
        tree = kdtree.build(DATA)
        self.assertTrue(tree.loc == [2, -2])
        
    def test_structure(self):
        tree = kdtree.build(DATA)
        self.assertIs(tree.left.left.left, None)
        self.assertIs(tree.left.left.right, None)
        self.assertIs(tree.left.right.left, None)
        self.assertIs(tree.left.right.right, None)
        self.assertIs(tree.right.left.left, None)
        self.assertIs(tree.right.left.right, None)
        self.assertIs(tree.right.right, None)

    def test_empty_points(self):
        tree = kdtree.build([])
        self.assertIs(tree, None)

    def test_single_point_data(self):
        tree = kdtree.build([[1, 2, 3]])
        self.assertEqual(tree.loc, [1, 2])
        self.assertIs(tree.left, None)
        self.assertIs(tree.right, None)

