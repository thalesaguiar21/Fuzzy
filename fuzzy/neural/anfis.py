from itertools import product

import numpy as np

from ..logic import subsets
from ..logic import mfs


class ANFIS:

    def __init__(self, fset_size=2, mepoch=100, tol=0.1):
        self.fset_size = fset_size
        self.max_epoch = mepoch
        self.tol = tol
        self.errors = [np.inf]
        self._prem = None
        self._cons = None
        self._npoints = 0
        self._nrules = 0
        self._nparams = 0
        self._ninputs = 0


    def fit(self, X, Y, strat='offline'):
        if strat == 'offline':
            self.fit_offline(X, Y)
        elif strat == 'online':
            self.fit_online(X, Y)
        else:
            raise ValueError(f'Unknown learning strategy: {strat}')

    def fit_online(self, X, Y):
        pass

    def fit_offline(self, X, Y):
        self._configure_network(X)
        epoch = 0
        while epoch < self.max_epoch and not self.has_converged():
            o5 = []
            for x, y in zip(X, Y):
                o1, o2, o3 = self.half_forward_pass(x)
                x_ = np.append(x, 1.0)
                fi = self._cons @ x_
                o4 = fi * o3
                o5.append(o4.sum())
    
    def _configure_network(self, X):
        self._npoints, self._ninputs = X.shape
        self._nmfs = self._ninputs * self.fset_size
        self._nrules = self.fset_size ** self._ninputs
        self._nparams = 2*self._nmfs + self._nrules*(self._ninputs + 1)
        self._prem = subsets.build_genbell(X, self._nmfs)

    def has_converged(self):
        return self.errors[-1] < self.tol

    def _get_mfids(self):
        np_mfids = np.arange(self._nmfs).reshape(self._ninputs, self.fset_size)
        return np_mfids.tolist()


    def set_params(self, **params):
        pass


class HybridOffline:

    def find_consequents(X, Y):
        ninputs, __ = data.shape
        for x in data:
            __, __, weights = half_forward_pass(x)
            weightcol = weigths.reshape((nrules, 1))
            x_ = np.append(x, 1).reshape((1, ninputs + 1))
            sys_products = weightcol @ x_
            sysline = sys_products.reshape(-1)
            sysmat.append(sysline)

        solution = lse.Matricial().solve(np.array(sysmat), Y)
        cons = np.reshape(solution, (nrules, ninputs + 1))
        return cons


    def half_forward_pass(x):
        a, b, c = self._prem[:, 0], self._prem[:, 1], self._prem[:, 2]
        l1 = mfs.genbell(np.repeat(x, self._nmfs), a, b, c)
        l2 = []
        for rule in product(*self._get_mfids()):
            prod = 1
            for mfid in rule:
                prod *= l1[mfid]
            l2.append(prod)
        total = np.sum(l2)
        l2 = np.array(l2)
        l3 = l2 / total
        return l1, l2, l3

