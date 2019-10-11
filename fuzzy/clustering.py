import numpy as np
import sklearn.decomposition as skdecomp
from .lse import Matricial
from .pca import PCA


class Clusterizer:

    def __init__(self):
        self.builders = {}

    def register_builder(self, model, builder):
        self.builders[model] = builder

    def create(self, model, **kwargs):
        builder = self.builders[model]
        if builder is None:
            raise ValueError(f"Builder {builder} not registred!")
        return builder(**kwargs)


class FCM:

    def __init__(self, nclusters, fuzzyness):
        self.validate_params(nclusters, fuzzyness)
        self.partitions = []
        self.m = fuzzyness
        self.nclusters = nclusters
        self.npoints = 0
        self.centroids = []

    def validate_params(self, nclusters, fuzzyness):
        if fuzzyness < 1:
            raise ValueError('Cluster fuzzyness must be at least one')
        if nclusters < 1:
            raise ValueError('There must be at least one cluster')

    def fit(self, data, tolerance):
        self.npoints = data.shape[0]
        self.partitions = self._init_partitions()
        self.centroids = np.zeros((self.nclusters, data.shape[1]))
        error = np.inf
        while error > tolerance:
            self._update_centroids(data)
            new_partitions = self._update_partitions(data)
            error = np.linalg.norm(new_partitions - self.partitions)
            self.partitions = new_partitions
        return self.partitions, self.centroids

    def _init_partitions(self):
        partitions = np.zeros((self.npoints, self.nclusters))
        for part in partitions:
            clust = np.random.randint(0, self.nclusters)
            part[clust] = 1.0
        return partitions

    def _update_centroids(self, data):
       for j in range(self.nclusters):
           denom = np.array([w ** self.m for w in self.partitions[:, j]])
           num = np.array([dt * wm for dt, wm in zip(data, denom)])
           self.centroids[j] = num.sum(axis=0) / denom.sum()

    def _update_partitions(self, data):
        U = np.zeros((self.npoints, self.nclusters))
        for i in range(self.npoints):
            for j in range(self.nclusters):
                U[i, j] = self._make_memdegree(data[i], j)
        return U

    def predict(self, samples):
        """ Associate and classify a sample as the class with the highest member
        ship degree """
        mem_degrees = self.predict_fuzz(samples)
        return np.argmax(mem_degrees, axis=1)

    def predict_fuzz(self, samples):
        """ Compute the membership degree of a sample to every cluster """
        if samples is None or len(samples) == 0:
            return []
        mem_degrees = np.zeros((samples.shape[0], self.nclusters))
        for i in range(samples.shape[0]):
            for j in range(self.nclusters):
                mem_degrees[i, j] = self._make_memdegree(samples[i], j)
        return mem_degrees

    def _make_memdegree(self, sample, j):
        num = np.linalg.norm(sample - self.centroids[j])
        dists = [np.linalg.norm(sample - ck) for ck in self.centroids]
        norm_dists = (num / np.array(dists)) ** (2.0 / (self.m-1.0))
        mem_degree = 1.0 / norm_dists.sum()
        return mem_degree


def fcm_builder(nclusters, fuzzyness):
    validate_fcm_arguments()
    return FCM(nclusters, fuzzyness)


def validate_fcm_arguments(nclusters, fuzzyness):
    if ncluster < 2:
        raise ValueError('There must be at least two clusters')
    if fuzzyness <= 1:
        raise ValueError('Cluster fuzzyness must be greater than 1')


class FGMM:
    """ Distance based Fuzzy Gaussian Mixture Model """

    def __init__(self, ncomponents, epsilon):
        self.ncomponents = ncomponents
        self.epsilon = epsilon

    def fit(self, data, fuzzyness=2, tolerance=0.2):
        centre = np.zeros((self.ncomponents, data.shape[1]))
        cov = np.zeros((self.ncomponents, data.shape[1]))
        fcm = FCM(self.ncomponents, fuzzyness)
        partitions, _ = fcm.fit(data, tolerance)
        pca = PCA(n_components=data.shape[1])
        loglike = 3
        threshold = 2
        while loglike > threshold:
            mixture_weights = _compute_mixture_weights(partitions)
            for i in range(self.ncomponents):
                parts_i = partitions[:, i]
                ai, bi = _find_curve_parameters(parts_i, data, pca)
                ai = self.epsilon + 1
                if abs(ai) < self.epsilon:
                    centre[i], cov[i] = self._compute_as_conventional_gmm(
                        parts_i, data, fuzzyness)
                else:
                    centre[i], cov[i] = self._compute_as_bent_gmm(parts_i,
                                                                  data,
                                                                  fuzzyness,
                                                                  pca,
                                                                  bi)
            self._update_partitions(partitions, centre, data, fuzzyness)
            break

    def _compute_as_conventional_gmm(self, partitions, data, m):
        fuzz_parts = partitions ** m
        weighted_points = np.sum((fuzz_parts * data.T).T, axis=0)
        centre = weighted_points / np.sum(fuzz_parts)
        variances = (data - centre) ** 2.0
        weighted_variances = np.sum((fuzz_parts * variances.T).T, axis=0)
        covariances = weighted_variances / np.sum(fuzz_parts)
        return centre, covariances

    def _compute_as_bent_gmm(self, partitions, data, m, pca, bi):
        fuzzparts = partitions ** m
        weighted_points = np.sum((fuzzparts * data.T).T, axis=0)
        means = weighted_points / np.sum(fuzzparts)
        padded_bi = np.zeros((1, data.shape[1]))
        padded_bi[0, 1] = bi
        centres = means + 1.0/pca.eigenvalues @ padded_bi.T + pca.translation
        covariances = weighted_points / np.sum(fuzzparts)
        return centres, covariances

    def predict(self, samples):
        pass

    def predict_fuzzy(self, samples):
        pass

    def _update_partitions(self, partitions, centre, data, fuzzyness):
        for i in range(data.shape[0]):
            for j in range(centre.shape[0]):
                partitions[i, j] = self._make_memdegree(data[i], centre, j, fuzzyness)

    def _make_memdegree(self, sample, centre, j, fuzzyness):
        num = np.linalg.norm(sample - centre[j])
        dists = [np.linalg.norm(sample - ck) for ck in centre]
        norm_dists = (num / np.array(dists)) ** (2.0 / (fuzzyness-1.0))
        mem_degree = 1.0 / norm_dists.sum()
        return mem_degree


def _compute_mixture_weights(partitions):
    cluster_relevance = np.sum(partitions, axis=0)
    total_weight = np.sum(cluster_relevance)
    return cluster_relevance / total_weight


def _find_curve_parameters(partitions, data, pca):
    weighted_data = (partitions * data.T).T
    transformed_points = pca.fit_transform(weighted_data)
    coefs, result = _make_system_matrices(transformed_points)
    ai, bi = Matricial().solve(coefs, result)
    return ai, bi


def _make_system_matrices(points):
    squared_x = points[:, 0]
    y = points[:, 1]
    coefs = np.vstack((squared_x, np.ones(squared_x.size)))
    return coefs.T, y

