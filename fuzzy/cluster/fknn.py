from collections import defaultdict

import numpy as np

from . import kdtree


class FKNN:
    ''' A Fuzzy K-Nearest Neighbours algorithm

    Args:
        k: the number of neighbours
        p: the distance power fo Minkowski metric
    '''
    def __init__(self, k, p, m, tree=None):
        self.k = k
        self.p = p
        self.m = 2 if m < 2 else m
        self._tree = tree
        self._nclasses = 0

    def fit(self, X, Y):
        self._nclasses = len(np.unique(Y))
        self._tree = _organise_data(X, Y)

    def predict(self, x):
        neighbours = self._find_neighbours(x)
        mdegrees = []
        dists = []
        for neigh in neighbours:
            dist = np.abs(x - _points(neigh)) ** (self.p/(self.m-1))
            dists.append(dist.sum())
            mdegrees.append(self._compute_mdegrees(neigh))
        pred = np.dot(dists, mdegrees) / sum(dists)
        return pred

    def _compute_mdegrees(self, point):
        mdegrees = np.zeros(self._nclasses)
        neighbours = self._find_neighbours(point)
        for neigh in _labels(neighbours):
            mdegrees[neigh] += 1
        return self._mdegree(mdegrees, _labels(point))

    def _find_neighbours(self, x):
        return kdtree.find_neighbours(self._tree, x, self.k, self.p)

    def _mdegree(self, cls_count, cls):
        mdegree = (0.49/self.k) * cls_count
        mdegree[cls] += 0.51
        return mdegree


def _organise_data(X, Y):
    labeled_data = np.hstack((X, Y))
    return kdtree.build(labeled_data.tolist())


def _labels(x):
    if len(x.shape) == 1:
        return x[-1]
    return x[:, -1]


def _points(x):
    if len(x.shape) == 1:
        return x[:-1]
    return x[:, :-1]


def clip(x, lower, higher):
    return min(max(x, lower), higher)


def minkowski(a, b, p):
    ''' Computes the Minkowki distance between two points
    '''
    a, b = np.array(a), np.array(b)
    if a.shape != b.shape:
        raise ValueError('Cannot compute with different array dimensions')
    pow_dists = np.abs(a - b)**p
    return pow_dists.sum() ** (1.0/p)

