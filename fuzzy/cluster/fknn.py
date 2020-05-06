from collections import defaultdict

import numpy as np

from . import kdtree
from ..logic import mfs


class FKNN:
    """A Fuzzy K-Nearest Neighbours algorithm.

    Attributes:
        nnghbours (int): the number of neighbours.
        p (int): the distance power fo Minkowski metric.
        m (int): the weight of the distances used for prediction, from 2 to
            'inf'.
    """
    def __init__(self, nneighbours, p, m, tree=None):
        self.nneighbours = nneighbours
        self.p = p
        self.m = m
        self._tree = tree
        self._nclasses = 0

    def fit(self, X, Y):
        """Organise the training data

        Args:
            X (ndarray): the feature vector
            Y (ndarray): the labels
        """
        _check_properties(self)
        _check_train_data(X, Y, self.nneighbours)
        self._count_classes(Y)
        self._tree = _organise_data(X, Y)

    def _count_classes(self, Y):
        self._nclasses = len(np.unique(Y))
        if self._nclasses < 2:
            raise ValueError('there must be at least 2 unique labels')

    def predict(self, testdata):
        """A crisp prediction from the fuzzyfied inferences """
        fuzz_predictions = self.predict_fuzz(testdata)
        return np.argmax(fuzz_predictions, axis=1)

    def predict_fuzz(self, testdata):
        """A fuzzy prediction from given data

        Returns:
            fuzzpreds (2d-ndarray): the class membership degree of each point
                to every class.
        """
        fuzzpredictions = []
        for point in testdata:
            pred = self._predict_single(point)
            fuzzpredictions.append(pred)
        return np.array(fuzzpredictions)

    def _predict_single(self, point):
        neighbours = self._find_neighbours(point)
        mdegrees = []
        dists = []
        for neigh in neighbours:
            dist = np.abs(point - _points(neigh)) ** (self.p/(self.m-1))
            dists.append(dist.sum())
            mdegrees.append(self._compute_mdegrees(neigh))
        pred = np.dot(dists, mdegrees) / sum(dists)
        return pred

    def _compute_mdegrees(self, point):
        mdegrees = np.zeros(self._nclasses)
        neighbours = self._find_neighbours(point)
        for neigh in _labels(neighbours):
            mdegrees[neigh] += 1
        return mfs.neighbours(self.nneighbours, mdegrees, _labels(point))

    def _find_neighbours(self, x):
        if not self._tree:
            raise ValueError('model is not trained')
        return kdtree.find_neighbours(self._tree, x, self.nneighbours, self.p)

    def set_params(self, **params):
        self.nneighbours = params.get('nneighbours', self.nneighbours)
        self.m = params.get('m', self.m)
        self.p = params.get('p', self.p)


def _organise_data(X, Y):
    nrows, fdim = X.shape
    labeled_data = np.hstack((X, Y.reshape((nrows, 1))))
    return kdtree.build(labeled_data.tolist())


def _check_properties(fknn):
    if fknn.m < 2:
        raise ValueError('distance weight \'m\' >= 2')
    if fknn.p not in [1, 2, 3]:
        raise ValueError('distance metric must be in [1, 2, 3]')
    if fknn.nneighbours < 1:
        raise ValueError('number of neighbours must be at least 1')


def _check_train_data(X, Y, nneighbours):
    if X.shape[0] < nneighbours:
        raise ValueError('there must be at least {nneighbours} points')
    _check_all_labeled(X, Y)


def _check_all_labeled(X, Y):
    notnone = X is not None and Y is not None
    if notnone and X.shape[0] != Y.shape[0]:
        raise ValueError('different number of points and labels')


def _labels(x):
    if len(x.shape) == 1:
        return int(x[-1])
    return x[:, -1].astype(np.int32)


def _points(x):
    if len(x.shape) == 1:
        return x[:-1]
    return x[:, :-1]

