import numpy as np

from ..lse import Matricial
from ..pca import PCA
from . import metrics


class FCM:

    def __init__(self, nclusters, fuzzyness, tol=1e-2, max_iter=200, metric=2):
        self.partitions = []
        self.m = fuzzyness
        self.nclusters = nclusters
        self.npoints = 0
        self.centroids = []
        self.tol = tol
        self.max_iter = max_iter
        self.p = metric

    def fit(self, X, Y=[]):
        _validate(self.nclusters, self.m)
        self._initialise_parts_centre_points(X, Y)
        errors = np.inf
        cur_iter = 1
        while not self.has_converged(errors, cur_iter):
            self._update_centroids(X)
            new_partitions = self._update_partitions(X)
            errors = np.abs(new_partitions - self.partitions)
            self.partitions = new_partitions
            cur_iter += 1
        return self.partitions, self.centroids

    def _initialise_parts_centre_points(self, X, Y):
        self.npoints, dim = X.shape
        self.partitions = self._init_partitions(Y)
        self.centroids = np.zeros((self.nclusters, dim))

    def _init_partitions(self, Y):
        partitions = np.random.rand(self.npoints, self.nclusters)
        for i in range(self.npoints):
            partitions[i] = partitions[i] / partitions[i].sum()
        return partitions

    def has_converged(self, errors, cur_iter):
        return np.max(errors) <= self.tol or cur_iter >= self.max_iter

    def _update_centroids(self, data):
        fuzzied_parts = self.partitions ** self.m
        dt_sum = fuzzied_parts.T @ data
        self.centroids = dt_sum / fuzzied_parts.sum()

    def _update_partitions(self, data):
        dists = self._make_dists(data)
        nonzero_dist = np.fmax(dists, np.finfo(np.float64).eps)
        return self._make_memdegree(nonzero_dist)

    def _make_dists(self, data):
        dists = np.zeros((data.shape[0], self.nclusters))
        for i in range(data.shape[0]):
            dists[i] = metrics.minkowski(data[i], self.centroids, self.p)
        return dists

    def _make_memdegree(self, dists):
        mem_degrees = np.zeros((dists.shape[0], self.nclusters))
        for i in range(dists.shape[0]):
            for j in range(self.nclusters):
                xi_dist = dists[i, j] / dists[i, :]
                norm_xi_dist = xi_dist ** (2.0 / (self.m-1.0))
                mem_degrees[i, j] = 1.0 / norm_xi_dist.sum()
        return mem_degrees

    def predict(self, samples):
        """ Associate and classify a sample as the class with the highest member
        ship degree """
        mem_degrees = self.predict_fuzz(samples)
        return np.argmax(mem_degrees, axis=1)

    def predict_fuzz(self, samples):
        """ Compute the membership degree of a sample to every cluster """
        if samples is None or len(samples) == 0:
            return []
        return self._update_partitions(samples)

    def set_params(self, **params):
        self.nclusters = params.get('nclusters', self.nclusters)
        self.m = params.get('fuzzyness', self.m)
        self.tol = params.get('tol', self.tol)
        self.p = params.get('metric', self.p)


def _validate(nclusters, fuzzyness):
    if nclusters < 2:
        raise ValueError('There msut be at least 2 clusters!')
    if fuzzyness <= 1:
        raise ValueError('Cluster fuzzyness must be greater than 1!')


class SemiSupervisedFCM(FCM):

    def __init__(self, nclusters, fuzzyness, tol=1e-2, max_iter=200, metric=2):
        super().__init__(nclusters, fuzzyness, tol, max_iter, metric)
        self._labels = []

    def fit(self, X, Y):
        self._labels = np.unique(Y.astype(np.int32))
        self.nclusters = self._labels.size
        super().fit(X, Y)
        self._label_clusters(Y)
        return self.partitions, self.centroids

    def _init_partitions(self, Y):
        Y_ = Y.reshape(-1).astype(np.int32)
        partitions = np.zeros((self.npoints, self._labels.size))
        for i, y in enumerate(Y_):
            partitions[i, y] = 1
        return partitions

    def _label_clusters(self, Y):
        Y_ = Y.reshape(-1).astype(np.int32)
        counts = {c: [0] * self._labels.size for c in range(self.nclusters)}
        best_fits = np.argmax(self.partitions, axis=1)
        for i, y in enumerate(Y_):
            counts[best_fits[i]][y] += 1
        self._clust_labels = np.argmax(list(counts.values()), axis=1)

    def predict(self, data):
        clusters = super().predict(data)
        return self._clust_labels[clusters]


