import numpy as np
import sklearn.datasets as skdata


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
       self.centroids = np.zeros((self.nclusters, data.shape[1]))
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
        norm_dists = num / np.array(dists)
        mem_degree = 1.0 / (norm_dists.sum() ** (2.0 / (self.m-10)))
        return mem_degree



class FGMM:
    """ Distance based Fuzzy Gaussian Mixture Model """

    def __init__(self, ncomponents):
        self.ncomponents = ncomponents

    def fit(self, data, fuzzyness=2, tolerance=0.2):
        fcm = FCM(self.ncomponents, fuzzyness)
        partitions, _ = fcm.fit(data, tolerance)

    def predict(self, samples):
        pass

    def predict_fuzzy(self, samples):
        pass


if __name__ == '__main__':
    fgmm = FGMM(4)
    data, labels = skdata.make_blobs(500, 2, 3)
    fgmm.fit(data, fuzzyness=2, tolerance=0.2)
