import numpy as np

from .fcm import FCM
from ..lse import Matricial
from ..pca import PCA


class FGMM:
    """ Probability based Fuzzy Gaussian Mixture Model """

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

