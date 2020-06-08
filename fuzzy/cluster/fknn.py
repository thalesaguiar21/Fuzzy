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
        self._memberships = []

    def fit(self, X, Y):
        """Organise the training data

        Args:
            X (ndarray): the feature vector
            Y (ndarray): the labels
        """
        _check_properties(self)
        _check_train_data(X, Y, self.nneighbours)
        self._count_classes(Y)
        self._build_membership_matrix(Y)
        self._tree = _organise_data(X, Y)

    def _build_membership_matrix(self, Y):
        self._nclasses = len(np.unique(Y))
        self._memberships = np.zeros((self._nclasses, Y.size))
        for i, y in enumerate(Y[:, 0].astype(np.int32)):
            self._memberships[y, i] = 1

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
        neigh_vecs, neigh_idx = neighbours[:, :-1], neighbours[:, -1]
        sqr_dists = np.sum((-neigh_vecs + point) ** 2, axis=1)
        dists = np.sqrt(sqr_dists)
        inv_dist = 1 / dists ** (self.p/(self.m-1))
        w_dists = self._memberships[:, neigh_idx.astype(np.int32)] @ inv_dist
        pred = w_dists / inv_dist.sum()
        return pred

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
    indexes = np.arange(X.shape[0]).reshape(X.shape[0], 1)
    labeled_data = np.hstack((X, indexes))
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

