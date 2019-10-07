import numpy as np


class PCA:

    def __init__(self, n_components):
        self.eigenvalues = []
        self.eigenvectors = []
        self.components = []
        self.translation = []
        self.n_components = n_components

    def fit_transform(self, data):
        self.translation = np.mean(data.T, axis=1)
        centred = data - self.translation
        self.components = np.cov(centred.T)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.components)
        newdata = self.eigenvectors.T @ self.components.T
        return newdata.T

