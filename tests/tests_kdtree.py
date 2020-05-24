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


class TestsDist(unittest.TestCase):

    def setUp(self):
        self.x = [1, 2]
        self.y = [3, -1]
        self.p = 0

    def get_dist(self):
        return kdtree.calc_dist(self.x, self.y, self.p)

    def test_manhattan(self):
        self.p = 1
        dist = self.get_dist()
        self.assertEqual(dist, 5)

    def test_euclidean(self):
        self.p = 2
        dist = self.get_dist()
        self.assertAlmostEqual(dist, 13)

    def test_mink(self):
        self.p = 3
        dist = self.get_dist()
        self.assertEqual(dist, 35)

    def test_power_4(self):
        with self.assertRaises(ValueError):
            self.p = 4
            self.get_dist()
    
    def test_power_0(self):
        with self.assertRaises(ValueError):
            self.p = 0
            self.get_dist()

    def test_power_negative(self):
        with self.assertRaises(ValueError):
            self.p = -1
            self.get_dist()
    

class TestsFind(unittest.TestCase):

    def setUp(self):
        self.point = [1, 3]
        self.neigh = 2
        self.p = 1
        self.data = DATA

    def find(self):
        tree = kdtree.build(self.data)
        return kdtree.find_neighbours(tree, self.point, self.neigh, self.p)

    def test_find_2(self):
        self.neigh = 2
        neighbours = self.find()
        self.assertEqual(neighbours.shape, (2, 3))
        self.assertTrue(all(neighbours[1] == [2, 1, 0]))
        self.assertTrue(all(neighbours[0] == [0, 1, 0]))

    def test_find_negative(self):
        with self.assertRaises(ValueError):
            self.neigh = -3
            neighbours = self.find()
    
    def test_zero_neighbours(self):
        self.neigh = 0
        neighbours = self.find()
        self.assertEqual(neighbours.size, 0)

    def test_single_neighbour(self):
        self.neigh = 1
        neighbour = self.find()
        self.assertEqual(neighbour.shape, (1, 3))
        self.assertTrue(all(neighbour[0] == [2, 1, 0]))

    def test_more_neighbours_than_points(self):
        self.neigh = 7
        neighbours = self.find()
        self.assertEqual(neighbours.shape, (6, 3))

    def test_n_neighbours_equal_to_data(self):
        self.neigh = 6
        neighbours = self.find()
        self.assertEqual(neighbours.shape, (6, 3))

    def test_n_neighbours_almost_equal_to_data(self):
        self.neigh = 5
        neighbours = self.find()
        self.assertEqual(neighbours.shape, (5, 3))

    def test_single_neighbour_in_data(self):
        self.neigh = 1
        self.point = [15, -8]
        neighbour = self.find()
        self.assertTrue(all(neighbour[0] == [15, -8, 1]))

    def test_find_wrong_metric(self):
        for bmetric in [-2, -1, 0, 4, 5]:
            self.p = bmetric
            with self.assertRaises(ValueError):
                self.find()

    def test_find_point_with_wrong_dim(self):
        for p in [[0], [3, 1, 3, 4], [0, 2, 0]]:
            self.point = p
            with self.assertRaises(ValueError):
                self.find()


class TestsToData(unittest.TestCase):

    def test_single_node(self):
        node = kdtree.Node([1, 2], [0], None, None)
        dt = kdtree.to_data([node])
        self.assertTrue(np.all(dt == [[1, 2, 0]]))

    def test_empty_input(self):
        dt = kdtree.to_data([])
        self.assertEqual(dt.size, 0)

    def test_3_nodes(self):
        nodes = [kdtree.Node([1, 2], [0], None, None),
                 kdtree.Node([2, 2], [1], None, None),
                 kdtree.Node([3, -1], [2], None, None)]
        dt = kdtree.to_data(nodes)
        equalnodes = np.all(dt == [[1, 2, 0], [2, 2, 1], [3, -1, 2]])
        self.assertTrue(equalnodes)

