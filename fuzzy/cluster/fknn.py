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
        self._memberships = []
        self._tree = tree

    def fit(self, X, Y):
        labeled_data = np.hstack((X, Y)).tolist()
        self._tree = kdtree.build(labeled_data)

    def predict(self, x):
        neighbours = kdtree.find_neighbours(self._tree, x, self.k, self.p)
        pred = getmostfrequent(neighbours[:][-1])
        return pred


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

