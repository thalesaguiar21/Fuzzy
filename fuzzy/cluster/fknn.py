from collections import defaultdict

import numpy as np

from . import kdtree


class FKNN:
    ''' A Fuzzy K-Nearest Neighbours algorithm

    Args:
        k: the number of neighbours
        p: the distance power fo Minkowski metric
    '''
    def __init__(self, k, p, tree=None):
        self.k = k
        self.p = p
        self._mdegrees = []
        self._tree = tree

    def fit(self, X, Y):
        self._tree = _organise_data(X, Y)
        self._compute_mdegrees(X, Y)

    def _compute_mdegrees(self, X, Y):
        n_classes, n_samples = len(np.unique(Y)), Y.shape[0]
        self._mdegrees = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            neighs = kdtree.find_neighbours(self._tree, X[i], self.k, self.p)
            cls_count = np.zeros(n_classes)
            for lbl in _labels(neighs):
                cls_count[lbl] += 1
            self._mdegrees[i] = self._mdegree(cls_count, Y[i])

    def _mdegree(self, cls_count, cls):
        mdegree = (0.49/self.k) * cls_count
        mdegree[cls] += 0.51
        return mdegree

    def predict(self, x):
        return -1


def _organise_data(X, Y):
    labeled_data = np.hstack((X, Y))
    return kdtree.build(labeled_data.tolist())


def _labels(x):
    return x[:, -1]


def getmostfrequent(sequence):
    max_ = 0
    freq_label = None
    counters = defaultdict(int)
    for x in sequence:
        counters[x] += 1
    for x in counters:
        if counters[x] > max_:
            max_ = counters[x]
            freq_label = x
    return x




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

