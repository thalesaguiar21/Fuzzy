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

    def fit(self, X, Y, init_strat='complete'):
        """Organise the training data

        Args:
            X (ndarray): the feature vector
            Y (ndarray): the labels
            init_strat (str): the membership initialisation strategy. Can be
                one of [complete, means], defaults to complete
        """
        _check_properties(self)
        _check_train_data(X, Y, self.nneighbours)
        self._count_classes(Y)
        self._memberships = _membership_factory(init_strat, X, Y, self.p,
                                                self.nneighbours)
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
        neigh_vecs, neigh_idx = neighbours[:, :-1], neighbours[:, -1]
        sqr_dists = np.sum(np.abs(-neigh_vecs + point) ** self.p, axis=1)
        dists = sqr_dists ** (1/self.p)
        inv_dist = 1 / dists ** (-self.p/(self.m-1))
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
    indexes = np.arange(X.shape[0], dtype=np.int32).reshape(X.shape[0], 1)
    labeled_data = np.hstack((X, indexes))
    return kdtree.build(labeled_data.tolist())


def _check_properties(fknn):
    if fknn.m <= 1:
        raise ValueError('distance weight \'m\' > 1')
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


def _membership_factory(strat, X, Y, p, k):
    if strat == 'complete':
        return _build_mdegs_complete(Y)
    elif strat == 'means':
        return _build_mdegs_kmeans(X, Y, p)
    elif strat == 'knn':
        return _build_mdegs_with_neighbours(X, Y, k, p)
    else:
        raise ValueError(f"Invalid strategy {strat}: use '[complete, means]'")


def _build_mdegs_complete(Y):
    nclasses = len(np.unique(Y))
    memberships = np.zeros((nclasses, Y.size))
    for j, y in enumerate(Y):
        memberships[y, j] = 1
    return memberships


def _build_mdegs_kmeans(X, Y, p):
    classes = np.unique(Y)

    centres = []
    for cls in classes:
        idxs = np.where(Y == cls)
        centres.append(np.mean(X[idxs], axis=0))
    centres = np.array(centres)

    memberships = np.zeros((len(classes), Y.size))
    for j, x in enumerate(X):
        pdist = np.sum(np.abs(-centres + x) ** p, axis=1)
        inv_dists = 1 / (pdist ** (1/p) + 1e-10)
        memberships[:, j] = inv_dists / inv_dists.sum()
    return memberships

def _build_mdegs_with_neighbours(X, Y, k, p):
    tree = _organise_data(X, Y)
    classes = np.unique(Y)
    memberships = np.zeros((len(classes), Y.size))
    for j, x in enumerate(X):
        neighbours = kdtree.find_neighbours(tree, x, k, p)
        neigh_vec = neighbours[:, :-1]
        neigh_idx = neighbours[:, -1].astype(np.int32)
        for y in Y[neigh_idx]:
            memberships[y, j] += 1
        memberships[:, j]  = (memberships[:, j] / k) * 0.49
        memberships[Y[j], j] += 0.51
    return memberships




