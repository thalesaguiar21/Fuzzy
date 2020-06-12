import numpy as np
from ..lse import Matricial
from ..pca import PCA


class FCM:

    def __init__(self, nclusters, fuzzyness, tol=1e-2, max_iter=200):
        self.partitions = []
        self.m = fuzzyness
        self.nclusters = nclusters
        self.npoints = 0
        self.centroids = []
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, Y):
        _validate(self.nclusters, self.m)
        self._initialise_parts_centre_points(X)
        errors = np.inf
        cur_iter = 1
        while not self.has_converged(errors, cur_iter):
            self._update_centroids(X)
            new_partitions = self._update_partitions(X)
            errors = np.abs(new_partitions - self.partitions)
            self.partitions = new_partitions
            cur_iter += 1
        return self.partitions, self.centroids

    def _initialise_parts_centre_points(self, data):
        self.npoints = data.shape[0]
        self.partitions = self._init_partitions()
        self.centroids = np.zeros((self.nclusters, data.shape[1]))

    def _init_partitions(self):
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
            sqr_dists = np.sum((-self.centroids + data[i]) ** 2, axis=1)
            dists[i] = np.sqrt(sqr_dists)
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


def _validate(nclusters, fuzzyness):
    if nclusters < 2:
        raise ValueError('There msut be at least 2 clusters!')
    if fuzzyness <= 1:
        raise ValueError('Cluster fuzzyness must be greater than 1!')

