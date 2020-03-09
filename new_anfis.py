from itertools import product

from sklearn.datasets import make_blobs
from sklearn.preprocessing import normalize
import numpy as np

from fuzzy import lse


class Anfis:

    def __init__(self, fsize, max_epoch):
        self.max_epoch = max_epoch
        self.n_mfs = 0
        self.n_rules = 0
        self.fsize = fsize
        self.n_inputs = 0
        self.n_points = 0
        self.l1 = []
        self.l2 = []
        self.l3 = []
        self.l4 = []
        self.premise = []
        self.consequents = []
        self.mfids = []
        self.Xa = []

    def fit(self, X, Y):
        print('Configuring network...')
        self._configure_layers(X)
        self._configure_premises()
        self._find_consequents(X, Y)
        self._fit(X, Y)

    def _configure_layers(self, X):
        self.n_points, self.n_inputs = X.shape
        self.n_mfs = self.n_inputs * self.fsize
        self.n_rules = self.fsize ** self.n_inputs
        self.mfids = np.arange(self.n_mfs).reshape(self.n_inputs, self.fsize)
        self.Xa = np.hstack((X, np.ones((self.n_points, 1))))

    def _configure_premises(self):
        mean = np.random.normal(0, 1.0, (self.n_mfs, 1))
        std = np.ones((self.n_mfs, 1))
        self.premise = np.hstack((mean, std))

    def _find_consequents(self, X, Y):
        print('Finding consequents...')
        linsys = []
        for x, y, x_ in zip(X, Y, self.Xa):
            self._half_forward_pass(x)
            weights = np.repeat(self.l3, self.n_inputs + 1)
            linsys.append(weights * x_)
        lse_result = lse.Matricial().solve(np.array(linsys), Y)
        conseq_shape = (self.n_points, self.n_inputs + 1)
        self.consequents = np.reshape(lse_result, conseq_shape)

    def _half_forward_pass(self, x):
        inps = np.repeat(x, self.fsize)
        args = (inps - self.premise[:, 0])/self.premise[:, 1]
        self.l1 = np.exp(-args)
        products = []
        for rule in product(*self.mfids.tolist()):
            prod = 1.0
            for mfid in rule:
                prod *= self.l1[mfid]
            products.append(prod)
        self.l2 = np.array(products)
        self.l3 = self.l2 / self.l2.sum()

    def _fit(self, X, Y):
        print('Training...')
        epoch = 0
        while epoch < self.max_epoch:
            print(f"Epoch {epoch + 1}")
            for x, y, x_ in zip(X, Y, self.Xa):
                self._forward_pass(x, x_)
                self._backward_pass(x_, y)
            epoch += 1

    def _forward_pass(self, x, x_):
        self._half_forward_pass(x)
        self.l4 = (consequents @ x_) * self.l3
        self.l5 = l4.sum()

    def _backward_pass(self, x_, y):
        dEdO5 = -2.0*(self.l5 - y)
        dO5dO4 = 1.0
        dO4dO3 = self.l4 / self.l3
        dO3dO2 = (-self.l2+self.l2.sum()) / self.l2.sum()**2
        dO2dO1 = []
        for k, rule in enumerate(product(*self.mfdids.tolist())):
            dOr = []
            for mfid in rule:
                dOr.append(self.l2[k] / self.l1[mfid])
            dO2dO1.append(dOr)
        dO2dO1 = np.array(dO2dO1)

        dO4dOc = np.tile(l3, (3, 1)).T * x_
        square_con_grad = dO4dOc ** 2
        eta_c = self.step / math.sqrt(square_con_grad.sum())
        self.consequents += eta_c * dO4dOc

if __name__ == '__main__':
    points, labels = make_blobs()
    X = normalize(points, axis=1, norm='l1')
    anfis = Anfis(10, 10)
    anfis.fit(X, labels)

