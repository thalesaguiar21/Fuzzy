from collections import defaultdict

import numpy as np

from . import kdtree


class FKNN:
    """ A Fuzzy K-Nearest Neighbours algorithm.

    Attributes:
        nnghbours (int): the number of neighbours.
        p (int): the distance power fo Minkowski metric.
        m (int): the weight of the distances used for prediction, from 2 to
            'inf'.
    """
    def __init__(self, nneighbours, p, m, tree=None):
        self.nneighbours = nneighbours
        self.p = p
        self.m = 2 if m < 2 else m
        self._tree = tree
        self._nclasses = 0

    def fit(self, X, Y):
        """ Organise the training data

        Args:
            X (ndarray): the feature vector
            Y (ndarray): the labels
        """
        self._validate_data(X, Y)
        self._count_classes(Y)
        self._tree = _organise_data(X, Y)

    def _validate_data(self, X, Y):
        if X is None or Y is None:
            raise ValueError(f'Invalid type of data {X} and {Y}')
        if X.shape[0] < self.nneighbours:
            raise ValueError('Data has less points than K')
        if X.shape[0] != Y.shape[0]:
            raise ValueError('Insuficient number of labels')

    def _count_classes(self, Y):
        self._nclasses = len(np.unique(Y))
        if self._nclasses < 2:
            raise ValueError('There must be at least 2 unique labels')

    def predict(self, X):
        if not isinstance(X[0], (list, np.ndarray)):
            X = list([X])
        predictions = []
        for x in X:
            pred = self._predict_single(x)
            predictions.append(pred)
        return predictions

    def _predict_single(self, x):
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
        return kdtree.find_neighbours(self._tree, x, self.nneighbours, self.p)

    def _mdegree(self, cls_count, cls):
        mdegree = (0.49/self.nneighbours) * cls_count
        mdegree[cls] += 0.51
        return mdegree


def _organise_data(X, Y):
    labeled_data = np.hstack((X, Y))
    return kdtree.build(labeled_data.tolist())


def _labels(x):
    if len(x.shape) == 1:
        return int(x[-1])
    return x[:, -1].astype(np.int32)


def _points(x):
    if len(x.shape) == 1:
        return x[:-1]
    return x[:, :-1]


