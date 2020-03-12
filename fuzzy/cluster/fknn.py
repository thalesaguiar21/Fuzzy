from dataclasses import dataclass

import numpy as np


class FKNN:

    def __init__(self, k, p, kdtree=None):
        self.k = k
        self.p = p
        self.memberships = []
        slef.kdtree = kdtree

    def fit(self, X, Y):
        pass

    def find_neighbours(self, x):
        pass



def clip(x, lower, higher):
    return min(max(x, lower), higher)


def minkowski(a, b, p):
    ''' Computes the Minkowki distance between two points
    '''
    a, b = np.array(a), np.array(b)
    if a.shape != b.shape:
        raise ValueError('Cannot compute with different array dimensions')
    pow_dists = np.abs(a - b)**p
    return np.abs(pow_dists).sum() ** (1.0/p)

breakpoint()
